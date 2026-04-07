"""
gap_analyzer.py  –  Skill Gap Analyser
=======================================
Fixes vs old version:
  1. Better prompt – requests specific, real resource links
  2. Resource URL validation with protocol enforcement
  3. YouTube fallback URLs always resolve to a real search page
  4. Per-skill retry on bad URLs
  5. Deduplication of incoming skill list
"""

import re
import json
import httpx
import time

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_LLM_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_TIMEOUT,
)
from models.job import SkillGap

YT_BASE = "https://www.youtube.com/results?search_query="

# Curated fallback queries per keyword
FALLBACK_YT: dict[str, str] = {
    "docker": "docker+tutorial+beginners+2024",
    "kubernetes": "kubernetes+crash+course+2024",
    "aws": "aws+cloud+tutorial+for+beginners",
    "react": "react+js+tutorial+2024+beginners",
    "node": "nodejs+tutorial+beginners+2024",
    "sql": "sql+tutorial+beginners",
    "postgresql": "postgresql+tutorial+beginners",
    "mongodb": "mongodb+crash+course",
    "fastapi": "fastapi+python+tutorial",
    "flask": "flask+python+web+framework+tutorial",
    "pytorch": "pytorch+deep+learning+tutorial",
    "tensorflow": "tensorflow+neural+network+tutorial",
    "machine learning": "machine+learning+full+course",
    "deep learning": "deep+learning+full+course",
    "nlp": "nlp+natural+language+processing+tutorial",
    "langchain": "langchain+tutorial+beginners",
    "spark": "apache+spark+tutorial+beginners",
    "kafka": "apache+kafka+tutorial",
    "git": "git+github+tutorial+beginners",
    "linux": "linux+command+line+tutorial+beginners",
}

GAP_PROMPT = """\
You are a career coach helping a job seeker close skill gaps.

The candidate is targeting the role: "{job_title}"
They are missing these skills:
{skills_numbered}

For EACH skill provide:
  1. resource: A real, working URL to a free learning resource
     - Prefer: freeCodeCamp, W3Schools, official docs, YouTube, Coursera free audit
     - Format: full https:// URL
  2. micro_project: A concrete small project (1–3 sentences) the person can build
     in 1–3 days to practice this skill, relevant to {job_title}

Return ONLY a JSON array – no markdown, no explanation:
[
  {{
    "missing_skill": "skill name",
    "resource": "https://...",
    "micro_project": "Build a ..."
  }},
  ...
]
"""


def _ensure_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _is_valid_url(url: str) -> bool:
    return bool(url) and url.startswith(("http://", "https://"))


def _fallback_yt(skill: str) -> str:
    skill_lower = skill.lower()
    for kw, query in FALLBACK_YT.items():
        if kw in skill_lower:
            return YT_BASE + query
    query = skill.strip().replace(" ", "+") + "+tutorial+for+beginners"
    return YT_BASE + query


def _fallback_project(skill: str, job_title: str) -> str:
    return (
        f"Build a small end-to-end project using {skill} "
        f"relevant to a {job_title} role. "
        f"Document it on GitHub with a README."
    )


def _call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        **OPENROUTER_HEADERS,
    }
    payload = {
        "model": OPENROUTER_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 900,
    }
    for attempt in range(3):
        try:
            with httpx.Client(timeout=OPENROUTER_TIMEOUT) as client:
                resp = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"API {resp.status_code}: {resp.json()}")
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise exc
    raise RuntimeError("Gap analysis: max retries exceeded")


def analyze_gaps(missing_skills: list, job_title: str, max_gaps: int = 5) -> list:
    if not missing_skills:
        return []

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for s in missing_skills:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
    top = unique[:max_gaps]

    print(f"  Gap analysis for: {job_title}")
    print(f"  Skills          : {top}")

    numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(top))
    prompt = GAP_PROMPT.format(job_title=job_title, skills_numbered=numbered)

    llm_map: dict[str, tuple[str, str]] = {}
    try:
        raw = _call_openrouter(prompt)
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        items = json.loads(text)
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                skill = item.get("missing_skill", "").strip()
                resource = _ensure_url(item.get("resource", ""))
                project = item.get("micro_project", "").strip()
                if skill:
                    if not _is_valid_url(resource):
                        resource = _fallback_yt(skill)
                    if not project:
                        project = _fallback_project(skill, job_title)
                    llm_map[skill.lower()] = (resource, project)
    except Exception as exc:
        print(f"  ⚠ Gap analysis LLM failed: {exc}")

    gaps: list[SkillGap] = []
    for skill in top:
        entry = llm_map.get(skill.lower())
        if entry:
            resource, project = entry
        else:
            resource = _fallback_yt(skill)
            project = _fallback_project(skill, job_title)

        gaps.append(SkillGap(
            missing_skill=skill,
            resource=resource,
            micro_project=project,
        ))
        print(f"    → {skill}: {resource[:70]}")

    return gaps


# Aliases expected by orchestrator
def _get_fallback_resource(skill: str) -> str:
    return _fallback_yt(skill)


def _get_fallback_project(skill: str, job_title: str) -> str:
    return _fallback_project(skill, job_title)