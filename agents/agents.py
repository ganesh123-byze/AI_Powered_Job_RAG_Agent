"""
agents.py  –  OpenRouter-based Agent Classes
=============================================
Fixes vs old version:
  1. ResumeParseAgent: delegates to resume_parser.py (single source of truth)
  2. SkillExtractorAgent: improved prompt + heuristic fallback aligned with
     the skill bank in resume_parser.py
  3. GapAnalysisAgent: delegates to gap_analyzer.py
  4. ResumeRewriterAgent: improved prompt with STAR format guidance
  5. All agents share one retry-aware LLM caller
"""

import json
import re
from typing import List, Dict, Optional

import httpx
import time

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_LLM_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class OpenRouterAgent:
    """Retry-aware base agent for OpenRouter API calls."""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = OPENROUTER_LLM_MODEL
        self.base_url = OPENROUTER_BASE_URL
        self.timeout = OPENROUTER_TIMEOUT
        self.max_retries = 3
        self.retry_delay = 2

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

    def call_llm(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        system: Optional[str] = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **OPENROUTER_HEADERS,
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]

                if resp.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"⚠ Rate limited. Waiting {wait}s…")
                    time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    raise ValueError("Invalid OpenRouter API key")
                if resp.status_code == 402:
                    raise ValueError("Insufficient OpenRouter credits")

                err = resp.json().get("error", {}).get("message", "Unknown")
                raise RuntimeError(f"OpenRouter {resp.status_code}: {err}")

            except (httpx.TimeoutException, RuntimeError) as exc:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"⚠ {exc}. Retry in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("LLM call failed after max retries")

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Resume parse agent  –  thin wrapper around resume_parser.py
# ---------------------------------------------------------------------------

class ResumeParseAgent(OpenRouterAgent):
    """Parses resume text into structured data."""

    def parse_resume(self, resume_text: str) -> Dict:
        """
        Call LLM with a comprehensive extraction prompt.
        Falls back to minimal dict on JSON failure.
        """
        prompt = (
            "You are an expert resume parser. Extract ALL information from "
            "the resume below and return ONLY valid JSON.\n\n"
            "CRITICAL RULES:\n"
            "- skills: list EVERY technology, tool, library, framework "
            "mentioned ANYWHERE in the resume (projects, experience, certs)\n"
            "- preferred_roles: infer 2-4 target job roles from skills/exp\n"
            "- experience_years: 0.0 if student/fresher\n"
            "- Return ONLY the JSON object, no markdown, no explanation\n\n"
            "Schema:\n"
            '{"full_name":str,"email":str|null,"phone":str|null,'
            '"skills":[str],"experience_years":float,"current_role":str|null,'
            '"education":[{"degree":str,"field_of_study":str,'
            '"institution":str,"year_of_graduation":int|null}],'
            '"projects":[{"name":str,"description":str,"technologies":[str]}],'
            '"preferred_roles":[str],"summary":str|null}\n\n'
            f"RESUME:\n{resume_text[:4000]}\n\nReturn ONLY JSON."
        )

        raw = self.call_llm(prompt, temperature=0.1, max_tokens=1800)
        text = self._clean_json(raw)

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # Try extracting first JSON object
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass

        print("⚠ ResumeParseAgent: JSON parse failed; returning minimal dict")
        return {
            "full_name": "Unknown",
            "email": None, "phone": None,
            "skills": [], "experience_years": 0.0,
            "current_role": None, "education": [],
            "projects": [], "preferred_roles": [],
            "summary": None,
        }


# ---------------------------------------------------------------------------
# Skill extractor agent
# ---------------------------------------------------------------------------

class SkillExtractorAgent(OpenRouterAgent):
    """Extracts technical skills from job descriptions."""

    def extract_skills(self, job_description: str) -> List[str]:
        prompt = (
            "Extract ALL technical skills, tools, frameworks, and technologies "
            "from the job description below.\n"
            "Include: programming languages, databases, cloud platforms, "
            "frameworks, libraries, DevOps tools, ML frameworks.\n"
            "Exclude: soft skills, company benefits, general terms.\n"
            "Return ONLY a JSON array of short strings (1-3 words each).\n"
            'Example: ["Python", "PostgreSQL", "Docker", "Machine Learning"]\n\n'
            f"Job description:\n{job_description[:2500]}\n\n"
            "Return ONLY the JSON array."
        )

        raw = self.call_llm(prompt, temperature=0.1, max_tokens=500)
        text = self._clean_json(raw)

        try:
            skills = json.loads(text)
            if isinstance(skills, list):
                result = [s.strip() for s in skills if isinstance(s, str) and s.strip()]
                if result:
                    return result
        except json.JSONDecodeError:
            pass

        return self._fallback_extract(job_description)

    def _fallback_extract(self, jd: str) -> List[str]:
        """Keyword-bank heuristic fallback."""
        from agents.resume_parser import SKILL_BANK  # reuse same bank
        jd_lower = jd.lower()
        found: list[str] = []
        seen: set[str] = set()
        for skill in SKILL_BANK:
            if re.search(r"\b" + re.escape(skill) + r"\b", jd_lower):
                if skill not in seen:
                    seen.add(skill)
                    found.append(skill.title())
        return found[:25]


# ---------------------------------------------------------------------------
# Gap analysis agent
# ---------------------------------------------------------------------------

class GapAnalysisAgent(OpenRouterAgent):
    """Analyses skill gaps and provides resources + micro-projects."""

    def analyze_gaps(self, missing_skills: List[str], job_title: str) -> List[Dict]:
        """Delegates to gap_analyzer.py for consistent behaviour."""
        from agents.gap_analyzer import analyze_gaps
        return analyze_gaps(missing_skills, job_title)


# ---------------------------------------------------------------------------
# Resume rewriter agent
# ---------------------------------------------------------------------------

class ResumeRewriterAgent(OpenRouterAgent):
    """Rewrites resume bullets to match a target job description."""

    def rewrite_bullets(
        self,
        current_bullets: List[str],
        job_title: str,
        matched_skills: List[str],
        missing_skills: List[str],
    ) -> List[Dict]:
        bullets_str = "\n".join(f"- {b}" for b in current_bullets[:6])
        matched_str = ", ".join(matched_skills[:6])

        prompt = (
            f"Rewrite the resume bullets below for the role: {job_title}\n\n"
            f"Current bullets:\n{bullets_str}\n\n"
            f"Skills to emphasise: {matched_str}\n\n"
            "Rules:\n"
            "1. Use STAR format (Situation → Task → Action → Result)\n"
            "2. Add quantifiable metrics where plausible (%, x improvement, etc.)\n"
            "3. Front-load the most relevant keyword for the role\n"
            "4. Keep each bullet under 25 words\n"
            "5. Do NOT invent new projects or companies\n\n"
            "Return ONLY a JSON array:\n"
            '[{"original":"...","rewritten":"...","skill_targeted":"..."}]\n\n'
            "Return ONLY the JSON array."
        )

        raw = self.call_llm(prompt, temperature=0.4, max_tokens=1200)
        text = self._clean_json(raw)

        try:
            rewrites = json.loads(text)
            if isinstance(rewrites, list):
                return rewrites
        except json.JSONDecodeError:
            pass
        return []


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_resume_parser: Optional[ResumeParseAgent] = None
_skill_extractor: Optional[SkillExtractorAgent] = None
_gap_analyzer: Optional[GapAnalysisAgent] = None
_resume_rewriter: Optional[ResumeRewriterAgent] = None


def get_resume_parser() -> ResumeParseAgent:
    global _resume_parser
    if _resume_parser is None:
        _resume_parser = ResumeParseAgent()
    return _resume_parser


def get_skill_extractor() -> SkillExtractorAgent:
    global _skill_extractor
    if _skill_extractor is None:
        _skill_extractor = SkillExtractorAgent()
    return _skill_extractor


def get_gap_analyzer() -> GapAnalysisAgent:
    global _gap_analyzer
    if _gap_analyzer is None:
        _gap_analyzer = GapAnalysisAgent()
    return _gap_analyzer


def get_resume_rewriter() -> ResumeRewriterAgent:
    global _resume_rewriter
    if _resume_rewriter is None:
        _resume_rewriter = ResumeRewriterAgent()
    return _resume_rewriter


# ---------------------------------------------------------------------------
# Convenience functions (backward-compatible API)
# ---------------------------------------------------------------------------

def extract_jd_skills(job_description: str) -> List[str]:
    return get_skill_extractor().extract_skills(job_description)


def analyze_gaps(missing_skills: List[str], job_title: str) -> List[Dict]:
    return get_gap_analyzer().analyze_gaps(missing_skills, job_title)


def rewrite_resume_for_job(
    profile,
    job_title: str,
    matched_skills: List[str],
    missing_skills: List[str],
) -> List[Dict]:
    rewriter = get_resume_rewriter()
    # Collect bullets from project descriptions and summary
    bullets: List[str] = []
    for proj in (profile.projects or []):
        if proj.description:
            bullets.append(proj.description[:120])
    if profile.summary:
        bullets.append(profile.summary[:120])
    if profile.current_role:
        bullets.append(f"Role: {profile.current_role}")
    return rewriter.rewrite_bullets(bullets, job_title, matched_skills, missing_skills)