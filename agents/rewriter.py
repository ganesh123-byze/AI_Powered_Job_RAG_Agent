"""
rewriter.py  –  Resume Bullet Rewriter
========================================
Improvements:
  1. _extract_resume_bullets() pulls from projects + experience + summary
  2. Prompt uses STAR format guidance
  3. Falls back gracefully with no crash
  4. Deduplicates bullets properly
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
from models.job import RewrittenBullet

REWRITE_PROMPT = """\
You are an expert resume coach. Rewrite the candidate's resume bullets
to better match the target job role.

CANDIDATE BULLETS:
{resume_bullets}

TARGET ROLE: {job_title}
MATCHED SKILLS (emphasise these): {matched_skills}
MISSING SKILLS (address if possible): {missing_skills}

RULES:
- Pick exactly 3 bullets from the list above
- Rewrite each in STAR format (Action → Task → Result with metric)
- Front-load the most relevant keyword for the role
- Keep each bullet under 25 words
- Stay truthful – rephrase/emphasise only, never invent experience
- Each rewrite should address a different matched or missing skill

Return ONLY a valid JSON array – no markdown, no explanation:
[
  {{
    "original": "exact original bullet text",
    "rewritten": "improved STAR-format bullet",
    "skill_targeted": "skill this highlights"
  }}
]"""


def _extract_resume_bullets(profile) -> list[str]:
    """Extract meaningful bullets from projects, summary, and skills."""
    bullets: list[str] = []

    # Project descriptions
    for proj in (profile.projects or []):
        if proj.description and len(proj.description.strip()) > 20:
            bullets.append(proj.description.strip()[:180])
        # Tech usage bullets
        if proj.technologies and proj.name:
            techs = ", ".join(proj.technologies[:4])
            bullets.append(f"Developed {proj.name} using {techs}")

    # Summary sentences
    if profile.summary:
        for sentence in re.split(r"[.!?]\s+", profile.summary):
            s = sentence.strip()
            if len(s) > 25:
                bullets.append(s[:180])

    # Skills summary (fallback)
    if profile.skills:
        top = ", ".join(profile.skills[:8])
        bullets.append(f"Proficient in {top}")

    # Deduplicate, min length 20
    seen: set[str] = set()
    unique: list[str] = []
    for b in bullets:
        key = b.lower().strip()
        if key not in seen and len(key) > 20:
            seen.add(key)
            unique.append(b)

    return unique[:12]


def _call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        **OPENROUTER_HEADERS,
    }
    payload = {
        "model": OPENROUTER_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 1000,
    }
    wait_times = [15, 30, 60]
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
                wait = wait_times[attempt]
                print(f"  Rewriter: rate limit – waiting {wait}s…")
                time.sleep(wait)
                continue
            raise RuntimeError(f"API {resp.status_code}: {resp.json()}")
        except RuntimeError:
            raise
        except Exception as exc:
            if attempt < 2:
                time.sleep(wait_times[attempt])
            else:
                raise RuntimeError(f"Rewriter call failed: {exc}") from exc
    raise RuntimeError("Rewriter: max retries exceeded")


def rewrite_resume_for_job(
    profile,
    job_title: str,
    matched_skills: list,
    missing_skills: list,
) -> list:
    """Rewrite top resume bullets to match the target job."""
    bullets = _extract_resume_bullets(profile)

    if not bullets:
        print(f"  Rewriter: no bullets found for {profile.full_name}")
        return []

    if not matched_skills and not missing_skills:
        print("  Rewriter: no skill context – skipping")
        return []

    prompt = REWRITE_PROMPT.format(
        resume_bullets="\n".join(f"- {b}" for b in bullets),
        job_title=job_title,
        matched_skills=", ".join(matched_skills[:6]) if matched_skills else "none",
        missing_skills=", ".join(missing_skills[:5]) if missing_skills else "none",
    )

    try:
        text = _call_openrouter(prompt)
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)

        rewrites: list[RewrittenBullet] = []
        for item in (data if isinstance(data, list) else []):
            if not isinstance(item, dict):
                continue
            original  = item.get("original", "").strip()
            rewritten = item.get("rewritten", "").strip()
            skill     = item.get("skill_targeted", "").strip()
            if original and rewritten and original != rewritten:
                rewrites.append(RewrittenBullet(
                    original=original,
                    rewritten=rewritten,
                    skill_targeted=skill,
                ))

        print(f"  Rewriter: {len(rewrites)} bullets rewritten for '{job_title}'")
        return rewrites[:3]

    except Exception as exc:
        print(f"  Rewriter: failed for '{job_title}' – {exc}")
        return []