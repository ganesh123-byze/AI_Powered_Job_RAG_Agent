"""
scorer.py  –  Hybrid Scorer with Multi-Signal Experience Inference
===================================================================
Key change vs old version:
  _score_experience() is replaced by exp_inference.infer_experience()
  which uses a 4-layer tiered system:
    1. Title tokens          (confidence 0.55–0.75)
    2. Adzuna snippet regex  (confidence 0.70)
    3. Full JD page fetch    (confidence 0.90–0.95)  — only when uncertain
    4. Confidence-weighted match score

  Jobs with evidence_source="none" / confidence < 0.25 get a neutral
  score of 0.50 instead of a hard rejection.

Everything else (skill scoring, location, freshness, hybrid retrieval)
is unchanged from the previous update.
"""

import re
from typing import Optional

from agents.agents import extract_jd_skills
from utils.skill_matcher import (
    skills_match, normalize_skill, get_skill_tokens, find_matched_skills,
)
from utils.exp_inference import infer_experience, clear_url_cache
from config import SCORE_WEIGHTS
from models.job import JobListing, MatchResult, ScoreBreakdown, SkillGap
from models.resume import ResumeProfile, UserPreferences
from utils.bm25_index import BM25IndexManager
from utils.hybrid_retriever import HybridRetriever
from utils.chroma_client import get_job_collection
from utils.embedder import embed_for_retrieval, build_resume_query_text
from utils.freshness import get_freshness_score

# ---------------------------------------------------------------------------
TOP_K_RETRIEVE = 50
TOP_K_RETURN   = 10
# ---------------------------------------------------------------------------


# ── Skill matching helpers ──────────────────────────────────────────────────

def _normalise(s: str) -> str:
    return normalize_skill(s)

def _skills_match(r: str, j: str) -> bool:
    return skills_match(r, j)

def _jd_skill_in_resume(jd: str, resume: list[str]) -> bool:
    return any(skills_match(jd, r) for r in resume)


# ── Skill overlap scoring ───────────────────────────────────────────────────

def _extract_jd_skills(jd_text: str) -> list[str]:
    try:
        skills = extract_jd_skills(jd_text)
        if skills:
            return skills
    except Exception:
        pass
    try:
        from agents.resume_parser import extract_skills_heuristically
        return extract_skills_heuristically(jd_text)
    except Exception:
        return []


def _score_skill_overlap(
    resume_skills: list[str],
    job_description: str,
) -> tuple[float, list[str], list[str]]:
    if not resume_skills:
        return 0.0, [], []

    jd_skills = _extract_jd_skills(job_description)

    if not jd_skills:
        jd_lower = job_description.lower()
        matched = [s for s in resume_skills
                   if _normalise(s) in jd_lower or s.lower() in jd_lower]
        score = len(matched) / max(len(resume_skills), 1)
        missing = [s for s in jd_skills if s not in matched]
        return round(min(score, 1.0), 4), matched, missing

    matched_resume, jd_gaps = find_matched_skills(resume_skills, jd_skills)

    jd_covered = len(jd_skills) - len(jd_gaps)
    score_a = jd_covered / max(len(jd_skills), 1)
    score_b = len(matched_resume) / max(len(resume_skills), 1)
    score   = 0.65 * score_a + 0.35 * score_b

    print(f"    Skills   : resume={len(resume_skills)}  JD={len(jd_skills)}")
    print(f"    Matched  : {matched_resume[:4] or 'None'}")
    print(f"    Gaps     : {jd_gaps[:3] or 'None'}")
    print(f"    SkillPct : {round(score*100)}%")

    return round(min(score, 1.0), 4), matched_resume, jd_gaps


# ── Location scoring ────────────────────────────────────────────────────────

_CITY_VARIANTS: dict[str, list[str]] = {
    "hyderabad": ["hyderabad", "hyd", "telangana", "secunderabad"],
    "bangalore": ["bangalore", "bengaluru", "banglore", "karnataka"],
    "mumbai":    ["mumbai", "bombay", "maharashtra"],
    "delhi":     ["delhi", "new delhi", "ncr", "gurgaon", "gurugram", "noida"],
    "chennai":   ["chennai", "madras", "tamil nadu"],
    "pune":      ["pune", "pimpri"],
    "kolkata":   ["kolkata", "calcutta"],
    "remote":    ["remote", "work from home", "wfh", "anywhere", "pan india"],
}

def _city_canonical(loc: str) -> str:
    ll = loc.lower().strip()
    for canonical, variants in _CITY_VARIANTS.items():
        if any(v in ll for v in variants):
            return canonical
    return ll

def _score_location(job_location: str, location_priority: list[str]) -> float:
    job_c = _city_canonical(job_location)
    user_remotes = {"remote", "work from home", "wfh"}
    user_wants_remote = any(l.lower().strip() in user_remotes for l in location_priority)
    if user_wants_remote and job_c == "remote":
        return 1.0
    for i, loc in enumerate(location_priority):
        if _city_canonical(loc) == job_c:
            return 1.0 if i == 0 else (0.75 if i == 1 else 0.5)
    if job_c == "remote":
        return 0.8
    return 0.2


# ── Overall score ───────────────────────────────────────────────────────────

def _compute_overall(breakdown: ScoreBreakdown) -> float:
    w = SCORE_WEIGHTS
    return round(min(
        breakdown.skill_overlap       * w.get("skill_overlap", 0.50)
        + breakdown.experience_alignment * w.get("experience_alignment", 0.20)
        + breakdown.location_match       * w.get("location_match", 0.15)
        + breakdown.freshness            * w.get("freshness", 0.15),
        1.0
    ), 4)

def _build_skill_gaps(missing: list[str]) -> list[SkillGap]:
    return [SkillGap(missing_skill=s, resource="", micro_project="") for s in missing]


# ── Main retrieval + scoring ────────────────────────────────────────────────

def retrieve_and_score_hybrid(
    profile: ResumeProfile,
    preferences: UserPreferences,
    use_rrf: bool = True,
    debug: bool = False,
) -> list[MatchResult]:
    """Full hybrid pipeline → returns top 10 unique results."""

    clear_url_cache()   # fresh URL cache per run

    print("\n[STEP] Initialising hybrid search…")
    chroma_collection = get_job_collection()
    all_jobs = chroma_collection.get()
    docs  = all_jobs.get("documents") or []
    metas = all_jobs.get("metadatas") or []

    if not docs:
        print("  ⚠ ChromaDB empty — no jobs to score.")
        return []

    bm25_manager = BM25IndexManager()
    bm25_manager.build_from_chroma(all_jobs)

    skill_count = len(profile.skills or [])
    bm25_w  = 0.6 if skill_count >= 8 else 0.4
    vector_w = 1.0 - bm25_w

    retriever = HybridRetriever(
        bm25_manager=bm25_manager,
        chroma_client=chroma_collection,
        embedder=None,
        bm25_weight=bm25_w,
        vector_weight=vector_w,
        rrf_k=60,
    )

    query_text      = build_resume_query_text(profile)
    query_embedding = embed_for_retrieval(query_text)

    hybrid_results = retriever.retrieve(
        query_text=query_text,
        query_embedding=query_embedding,
        top_k=TOP_K_RETRIEVE,
        use_rrf=use_rrf,
        debug=debug,
        skill_list=profile.skills,
    )

    if not hybrid_results:
        print("  ⚠ Hybrid retriever returned 0 results.")
        return []

    print(f"\n  Retrieved {len(hybrid_results)} candidates → scoring…")

    # Small pre-rank boost for title/company meta overlap
    def _meta_overlap(skills, text):
        if not skills: return 0.0
        tl = text.lower()
        return sum(1 for s in skills if s.lower() in tl) / len(skills)

    max_hybrid = max((r.hybrid_score for r in hybrid_results), default=1.0)
    for r in hybrid_results:
        meta = bm25_manager.get_job_metadata(r.job_index) or {}
        meta_text = f"{meta.get('title','')} {meta.get('company','')}"
        r.hybrid_score = round(r.hybrid_score + 0.08 * _meta_overlap(profile.skills, meta_text), 4)
    hybrid_results.sort(key=lambda r: r.hybrid_score, reverse=True)

    # ── Score each candidate ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SCORING {len(hybrid_results)} candidates")
    print(f"{'='*60}")

    # Determine if we should fetch full JDs (skip for obvious high-confidence cases)
    # Fetch only if we haven't already retrieved enough high-confidence results
    fetch_budget = 15    # max destination-page fetches per pipeline run
    fetch_count  = 0

    match_results: list[MatchResult] = []

    for i, hr in enumerate(hybrid_results, 1):
        job_idx   = hr.job_index
        full_desc = docs[job_idx] if job_idx < len(docs) else ""
        meta      = metas[job_idx] if job_idx < len(metas) else {}

        print(f"\n  [{i}/{len(hybrid_results)}] {hr.title} @ {hr.company}")
        print(f"    Hybrid: {hr.hybrid_score:.4f}  (BM25:{hr.bm25_score:.4f} Vec:{hr.vector_score:.4f})")

        # ── Skill score ───────────────────────────────────────────────
        skill_score, matched, missing = _score_skill_overlap(
            profile.skills, full_desc
        )

        # ── Experience score (4-layer inference) ──────────────────────
        # Decide whether to spend a fetch slot on this job
        do_fetch = fetch_count < fetch_budget

        exp_result = infer_experience(
            title=hr.title,
            snippet=full_desc[:1500],       # Adzuna snippet stored in chroma
            redirect_url=hr.apply_url,
            user_level=preferences.experience_level,
            candidate_years=profile.experience_years,
            fetch_full_jd=do_fetch,
        )

        if exp_result.signal.evidence_source in ("full_jd", "full_jd_phrase"):
            fetch_count += 1

        exp_score = exp_result.score

        print(f"    ExpInfer : label={exp_result.label}  conf={exp_result.confidence:.2f}"
              f"  match={exp_result.is_match}  src={exp_result.signal.evidence_source}")
        print(f"    ExpReason: {exp_result.reason[:80]}")

        # ── Skip hard mismatches with high confidence ─────────────────
        if not exp_result.is_match and exp_result.confidence >= 0.80:
            print(f"    ✗ Excluded — strong exp mismatch (conf={exp_result.confidence:.2f})")
            continue

        # ── Location + freshness ──────────────────────────────────────
        loc_score   = _score_location(hr.location, preferences.location_priority)
        fresh_score = get_freshness_score(hr.posted_date)

        breakdown = ScoreBreakdown(
            skill_overlap=skill_score,
            experience_alignment=exp_score,
            location_match=loc_score,
            freshness=fresh_score,
        )
        breakdown.overall = _compute_overall(breakdown)

        # Blend retrieval signal (25%)
        retrieval_norm = (hr.hybrid_score / max_hybrid) if max_hybrid > 0 else 0.0
        breakdown.overall = round(
            min(0.75 * breakdown.overall + 0.25 * retrieval_norm, 1.0), 4
        )

        # Skip truly irrelevant (zero skill overlap AND very low retrieval)
        meta_text = f"{hr.title} {hr.company} {hr.location}"
        if (skill_score <= 0.0
                and _meta_overlap(profile.skills, meta_text) == 0.0
                and hr.hybrid_score < 0.01):
            print(f"    ✗ Skipped — no skill/meta overlap")
            continue

        job = JobListing(
            job_id=hr.job_id,
            title=hr.title,
            company=hr.company,
            location=hr.location,
            description=full_desc,
            posted_date=hr.posted_date,
            apply_url=hr.apply_url,
            salary_min=hr.salary_min,
            salary_max=hr.salary_max,
            freshness_bucket=hr.freshness_bucket,
        )

        match_results.append(MatchResult(
            job=job,
            score=breakdown,
            matched_skills=matched,
            missing_skills=_build_skill_gaps(missing),
            explanation=(
                f"Hybrid:{hr.hybrid_score:.4f} | "
                f"Exp:{exp_result.label}(conf={exp_result.confidence:.2f},"
                f"src={exp_result.signal.evidence_source})"
            ),
        ))

    match_results.sort(key=lambda x: x.score.overall, reverse=True)
    top10 = match_results[:TOP_K_RETURN]

    print(f"\n{'='*60}")
    print(f"✅ {len(top10)} top jobs returned  (fetch_count={fetch_count})")
    print(f"{'='*60}")
    for i, r in enumerate(top10, 1):
        print(f"  {i:>2}. {r.job.title} @ {r.job.company}"
              f"  Overall:{round(r.score.overall*100)}%"
              f"  Skills:{round(r.score.skill_overlap*100)}%"
              f"  Exp:{round(r.score.experience_alignment*100)}%")

    return top10


def retrieve_and_score(
    profile: ResumeProfile,
    preferences: UserPreferences,
) -> list[MatchResult]:
    return retrieve_and_score_hybrid(profile, preferences, use_rrf=True, debug=False)