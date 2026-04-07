"""
job_fetcher.py  –  Smart Adzuna Job Fetcher
============================================
Fixes vs old version:
  1. Query strategy derived from skills directly – not just domain
  2. Skill-weighted queries: top skills become search keywords
  3. Role-based queries added automatically
  4. Improved relevance filter: partial / stemmed skill matching
  5. Graceful fallback: if strict filters yield 0 jobs, relax them
  6. Deduplication by job_id across all queries
  7. Clear progress output per query
"""

import re
import hashlib
import requests
from typing import Optional

from config import (
    ADZUNA_APP_ID,
    ADZUNA_API_KEY,
    ADZUNA_BASE_URL,
    ADZUNA_COUNTRY,
    ADZUNA_RESULTS_PER_PAGE,
)
from models.job import JobListing
from utils.freshness import get_freshness_bucket
from utils.chroma_client import upsert_job, clear_job_collection
from utils.embedder import embed_for_retrieval
from utils.skill_matcher import normalize_skill

# ---------------------------------------------------------------------------
# Experience-level filters (same as before, kept for compatibility)
# ---------------------------------------------------------------------------

TITLE_REJECT = {
    "fresher": {
        "senior", "sr", "principal", "staff", "architect",
        "lead", "leader", "head", "chief", "director", "vp",
        "manager", "expert", "specialist", "guru", "veteran",
    },
    "junior": {
        "senior", "sr", "principal", "staff", "architect",
        "lead", "head", "chief", "director", "vp", "manager",
        "expert", "guru",
    },
    "mid": {
        "principal", "staff", "chief", "director", "vp",
        "vice president", "guru", "veteran",
    },
    "senior": set(),
}

DESC_REJECT = {
    "fresher": {
        "4+ years", "5 years", "5+ years", "6 years", "6+ years",
        "7 years", "7+ years", "8 years", "8+ years", "10+ years",
        "minimum 4", "minimum 5", "at least 4", "at least 5",
    },
    "junior": {
        "5+ years", "6 years", "6+ years", "7 years", "7+ years",
        "8 years", "8+ years", "10+ years", "minimum 5", "at least 5",
    },
    "mid": {
        "10+ years", "12+ years", "15+ years",
    },
    "senior": set(),
}

EXP_LEVEL_RANGE = {
    "fresher": (0.0, 1.0),
    "junior": (0.0, 3.0),
    "mid": (2.0, 6.0),
    "senior": (4.0, 99.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_exp_range(desc: str) -> Optional[tuple[float, float]]:
    d = desc.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+\s*years?",
        r"(?:minimum|at\s+least|min\.?)\s+(\d+(?:\.\d+)?)\s*years?",
        r"(\d+(?:\.\d+)?)\s*years?\s*(?:of\s*)?(?:experience|exp)",
    ]
    for pat in patterns:
        m = re.search(pat, d)
        if m:
            groups = m.groups()
            lo = float(groups[0])
            hi = float(groups[1]) if len(groups) > 1 and groups[1] else 99.0
            return lo, hi
    return None


def _is_suitable_for_level(job: JobListing, exp_level: str) -> bool:
    title_lower = job.title.lower()
    desc_lower = job.description.lower()

    for kw in TITLE_REJECT.get(exp_level, set()):
        if re.search(r"\b" + re.escape(kw) + r"\b", title_lower):
            return False

    for kw in DESC_REJECT.get(exp_level, set()):
        if kw in desc_lower:
            return False

    exp_range = _extract_exp_range(desc_lower)
    if exp_range:
        jd_min, _ = exp_range
        lo, hi = EXP_LEVEL_RANGE.get(exp_level, (0, 99))
        if not (lo <= jd_min <= hi):
            return False

    return True


def _parse_job(raw: dict) -> JobListing:
    job_id = hashlib.md5(raw.get("redirect_url", "").encode()).hexdigest()
    return JobListing(
        job_id=job_id,
        title=raw.get("title", ""),
        company=raw.get("company", {}).get("display_name", ""),
        location=raw.get("location", {}).get("display_name", ""),
        description=raw.get("description", ""),
        posted_date=raw.get("created", "")[:10],
        apply_url=raw.get("redirect_url", ""),
        freshness_bucket=get_freshness_bucket(raw.get("created", "")[:10]),
    )


# ---------------------------------------------------------------------------
# Smart query builder
# ---------------------------------------------------------------------------

def _build_search_queries(profile) -> list[str]:
    """
    Build a prioritised list of Adzuna search queries from the resume profile.
    Priority order:
      1. Preferred role + top skill combinations
      2. Preferred roles alone
      3. Top individual skills
      4. Generic fallbacks
    """
    queries: list[str] = []
    skills = [s for s in (profile.skills or []) if len(s) > 1]
    roles = profile.preferred_roles or []

    # Deduplicate skills case-insensitively, prefer shorter (more searchable)
    seen: set[str] = set()
    top_skills: list[str] = []
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            top_skills.append(s)
    top_skills = top_skills[:20]

    # 1. Role + primary skill combos  e.g. "Machine Learning Engineer Python"
    for role in roles[:3]:
        for skill in top_skills[:4]:
            queries.append(f"{role} {skill}")

    # 2. Roles alone
    for role in roles[:4]:
        queries.append(role)

    # 3. Top skills paired (e.g. "Python Machine Learning")
    if len(top_skills) >= 2:
        for i in range(min(4, len(top_skills) - 1)):
            queries.append(f"{top_skills[i]} {top_skills[i+1]}")

    # 4. Individual top skills
    for skill in top_skills[:6]:
        queries.append(skill)

    # 5. Generic fallbacks
    queries.extend(["Data Scientist", "Software Engineer", "Python Developer"])

    # Deduplicate while preserving order
    seen_q: set[str] = set()
    final: list[str] = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen_q:
            seen_q.add(key)
            final.append(q)

    return final


def _normalise_locations(location_priority: list[str]) -> list[str]:
    """Expand common Indian city variations for Adzuna."""
    mapping = {
        "hyderabad": ["Hyderabad", "Telangana"],
        "hydrabad": ["Hyderabad", "Telangana"],
        "bangalore": ["Bangalore", "Bengaluru"],
        "bengaluru": ["Bangalore", "Bengaluru"],
        "mumbai": ["Mumbai", "Maharashtra"],
        "delhi": ["Delhi", "New Delhi"],
        "new delhi": ["Delhi", "New Delhi"],
        "chennai": ["Chennai", "Tamil Nadu"],
        "pune": ["Pune", "Maharashtra"],
        "kolkata": ["Kolkata"],
        "noida": ["Noida"],
        "gurgaon": ["Gurgaon", "Gurugram"],
    }
    out: list[str] = []
    for loc in location_priority:
        expanded = mapping.get(loc.lower())
        if expanded:
            out.extend(expanded)
        else:
            out.append(loc)
    # Deduplicate
    seen: set[str] = set()
    return [x for x in out if x.lower() not in seen and not seen.add(x.lower())]


# ---------------------------------------------------------------------------
# Skill relevance check  –  robust partial / normalised matching
# ---------------------------------------------------------------------------

_STEMMING = {
    # Map common variations to canonical forms
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["dl", "deep-learning"],
    "natural language processing": ["nlp"],
    "computer vision": ["cv"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "javascript": ["js"],
    "typescript": ["ts"],
    "nodejs": ["node.js", "node js", "node"],
    "postgresql": ["postgres"],
    "mongodb": ["mongo"],
    "kubernetes": ["k8s"],
    "github actions": ["github ci"],
}

# Build reverse map
_SKILL_VARIANTS: dict[str, list[str]] = {}
for canonical, variants in _STEMMING.items():
    for v in variants:
        _SKILL_VARIANTS[v] = [canonical] + variants
    _SKILL_VARIANTS[canonical] = [canonical] + variants


def _skill_in_text(skill: str, text_lower: str) -> bool:
    """Check if a skill (or its variants) appears in the text."""
    s = skill.strip().lower()
    if re.search(r"\b" + re.escape(s) + r"\b", text_lower):
        return True
    # Check normalised form
    normalised = normalize_skill(s).replace("_", " ")
    if re.search(r"\b" + re.escape(normalised) + r"\b", text_lower):
        return True
    # Check known variants
    for variant in _SKILL_VARIANTS.get(s, []):
        if re.search(r"\b" + re.escape(variant) + r"\b", text_lower):
            return True
    return False


def _count_matching_skills(profile_skills: list[str], job_text: str) -> tuple[int, list[str]]:
    """Return (match_count, matched_skill_names)."""
    job_lower = job_text.lower()
    matched = []
    for skill in profile_skills:
        if _skill_in_text(skill, job_lower):
            matched.append(skill)
    return len(matched), matched


# ---------------------------------------------------------------------------
# Adzuna API fetcher
# ---------------------------------------------------------------------------

def _fetch_adzuna_page(
    what: str,
    where: str,
    page: int,
) -> list[dict]:
    url = f"{ADZUNA_BASE_URL}/{ADZUNA_COUNTRY}/search/{page}"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "results_per_page": ADZUNA_RESULTS_PER_PAGE,
        "what": what,
        "where": where,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        return data.get("results", [])
    except Exception as exc:
        print(f"    ✗ Adzuna error ({what!r}, {where!r}, p{page}): {exc}")
        return []


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def fetch_and_store_jobs(profile, preferences) -> list[JobListing]:
    """
    Fetch jobs from Adzuna, filter, embed, and store in ChromaDB.
    Returns list of stored JobListing objects.
    """
    if not ADZUNA_APP_ID or not ADZUNA_API_KEY:
        print("WARN: Adzuna credentials missing – skipping fetch.")
        return []

    clear_job_collection()

    queries = _build_search_queries(profile)
    locations = _normalise_locations(preferences.location_priority or ["India"])

    print(f"\n{'='*60}")
    print("JOB FETCH STRATEGY")
    print(f"{'='*60}")
    print(f"  Locations : {locations}")
    print(f"  Queries   : {queries[:8]}")
    print(f"{'='*60}\n")

    stored_ids: set[str] = set()      # dedup across queries
    jobs: list[JobListing] = []

    target = 40          # desired number of stored jobs
    min_skill_match = 2  # minimum resume skills that must appear in JD
    strict = True        # start strict; relax if we get too few results

    for query in queries:
        if len(jobs) >= target:
            break

        for location in locations:
            if len(jobs) >= target:
                break

            for page in range(1, 3):
                results = _fetch_adzuna_page(query, location, page)
                if not results:
                    break

                print(f"  Query: {query!r} | {location} | p{page}"
                      f" → {len(results)} raw results")

                for raw in results:
                    job = _parse_job(raw)

                    # Skip already stored
                    if job.job_id in stored_ids:
                        continue

                    # Experience level filter
                    if not _is_suitable_for_level(job, preferences.experience_level):
                        continue

                    # Skill relevance filter
                    job_text = f"{job.title} {job.description}"
                    count, matched = _count_matching_skills(
                        profile.skills or [], job_text
                    )

                    threshold = min_skill_match if strict else 1
                    if count < threshold:
                        continue

                    # Passed all filters → embed and store
                    embed_text = f"{job.title} {job.company} {job.description}"
                    embedding = embed_for_retrieval(embed_text[:3000])

                    upsert_job(
                        job_id=job.job_id,
                        description=embed_text,
                        embedding=embedding,
                        metadata={
                            "job_id": job.job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "posted_date": job.posted_date,
                            "apply_url": job.apply_url,
                            "source": "adzuna",
                            "experience_level": preferences.experience_level,
                            "matched_skills": ",".join(matched[:10]),
                        },
                    )

                    stored_ids.add(job.job_id)
                    jobs.append(job)
                    print(f"    ✓ {job.title} @ {job.company}"
                          f"  skills:{matched[:3]}")

    # If strict mode returned too few results, retry with relaxed filter
    if len(jobs) < 5 and strict:
        print("\n  ⚠ Too few jobs with strict filter. Relaxing to 1 skill match...")
        strict = False
        for query in queries[:4]:
            if len(jobs) >= 20:
                break
            for location in locations[:2]:
                results = _fetch_adzuna_page(query, location, 1)
                for raw in results:
                    job = _parse_job(raw)
                    if job.job_id in stored_ids:
                        continue
                    job_text = f"{job.title} {job.description}"
                    count, matched = _count_matching_skills(
                        profile.skills or [], job_text
                    )
                    if count < 1:
                        continue
                    embed_text = f"{job.title} {job.company} {job.description}"
                    embedding = embed_for_retrieval(embed_text[:3000])
                    upsert_job(
                        job_id=job.job_id,
                        description=embed_text,
                        embedding=embedding,
                        metadata={
                            "job_id": job.job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "posted_date": job.posted_date,
                            "apply_url": job.apply_url,
                            "source": "adzuna",
                            "experience_level": preferences.experience_level,
                            "matched_skills": ",".join(matched[:10]),
                        },
                    )
                    stored_ids.add(job.job_id)
                    jobs.append(job)
                    print(f"    ✓ (relaxed) {job.title} @ {job.company}")

    print(f"\n  ✅ Total jobs stored: {len(jobs)}\n")
    return jobs