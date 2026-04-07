"""
orchestrator.py  –  JobRAG Pipeline Orchestrator
=================================================
Fixes vs old version:
  1. Step 3 scorer now gets job objects directly (no orphan IDs)
  2. Step 4 gap analysis only calls LLM for top 5; rest use fast fallback
  3. Step 5 rewriter uses project descriptions as bullets (not edu records)
  4. Cleaner summary table at end
  5. All Gemini imports removed
"""

import time
import logging
import traceback

from models.resume import ResumeProfile, UserPreferences
from models.job import MatchResult, SkillGap
from agents.resume_parser import parse_resume
from agents.job_fetcher import fetch_and_store_jobs
from agents.scorer import retrieve_and_score, _jd_skill_in_resume
from agents.gap_analyzer import analyze_gaps, _get_fallback_resource, _get_fallback_project
from agents.rewriter import rewrite_resume_for_job

logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_path: str,
    experience_level: str,
    freshness_days: int,
    location_priority: list[str],
) -> tuple[ResumeProfile, list[MatchResult]]:

    # ── STEP 1: Parse Resume ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 – Parsing Resume")
    print("=" * 60)
    try:
        profile = parse_resume(pdf_path)
    except Exception as exc:
        logger.error("Resume parsing failed: %s\n%s", exc, traceback.format_exc())
        raise RuntimeError(f"Resume parsing failed: {exc}") from exc

    print(f"\n✓ Resume parsed successfully")
    print(f"  Name        : {profile.full_name}")
    print(f"  Experience  : {profile.experience_years} yrs")
    print(f"  Roles       : {profile.preferred_roles}")
    print(f"  Skills      : {len(profile.skills)} found")
    if profile.skills:
        print(f"  Sample      : {profile.skills[:12]}")

    if not profile.skills:
        logger.warning("No skills extracted from resume. Job matching will be weak.")
        print("\n  ⚠ WARNING: No skills were extracted. "
              "Check PDF text extraction or prompt output.")

    preferences = UserPreferences(
        experience_level=experience_level,
        freshness_days=freshness_days,
        location_priority=location_priority,
    )

    # ── STEP 2: Fetch Jobs ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2 – Fetching Jobs from Adzuna")
    print("=" * 60)
    jobs = fetch_and_store_jobs(profile, preferences)
    print(f"  Total jobs stored: {len(jobs)}")

    if not jobs:
        print("  ⚠ No jobs fetched. Check Adzuna credentials and network.")
        return profile, []

    # ── STEP 3: Score and Rank ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3 – Scoring and Ranking")
    print("=" * 60)
    results = retrieve_and_score(profile, preferences)
    print(f"  Top {len(results)} matches scored")

    if not results:
        print("  ⚠ No results after scoring. Returning empty.")
        return profile, []

    # ── STEP 4: Gap Analysis ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 4 – Skill Gap Analysis")
    print("=" * 60)

    for i, match in enumerate(results):
        # Only include gaps for skills truly missing from resume
        verified = [
            g.missing_skill
            for g in match.missing_skills
            if not _jd_skill_in_resume(g.missing_skill, profile.skills)
        ]

        if not verified:
            match.missing_skills = []
            print(f"  [{i+1}] No gaps – {match.job.title}")
            continue

        if i < 5:
            # Full LLM gap analysis for top 5
            print(f"  [{i+1}] Enriching {len(verified)} gaps – {match.job.title}")
            enriched = analyze_gaps(verified, match.job.title, max_gaps=5)
            match.missing_skills = enriched
            time.sleep(0.8)
        else:
            # Fast fallback for the rest
            match.missing_skills = [
                SkillGap(
                    missing_skill=skill,
                    resource=_get_fallback_resource(skill),
                    micro_project=_get_fallback_project(skill, match.job.title),
                )
                for skill in verified[:5]
            ]
            print(f"  [{i+1}] Fallback gaps – {match.job.title}")

    # ── STEP 5: Resume Rewrite for Top 3 ─────────────────────────────────
    time.sleep(2)
    print()
    print("=" * 60)
    print("STEP 5 – Resume Bullet Rewriting (Top 3 Jobs)")
    print("=" * 60)

    for i, match in enumerate(results[:3]):
        missing_names = [g.missing_skill for g in match.missing_skills]
        print(f"  [{i+1}] Rewriting for: {match.job.title}")
        try:
            rewrites = rewrite_resume_for_job(
                profile=profile,
                job_title=match.job.title,
                matched_skills=match.matched_skills,
                missing_skills=missing_names,
            )
            match.rewrites = rewrites
        except Exception as exc:
            logger.warning("Rewrite failed for %s: %s", match.job.title, exc)
            match.rewrites = []
        time.sleep(3)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Candidate   : {profile.full_name}")
    print(f"  Jobs fetched: {len(jobs)}")
    print(f"  Top matches : {len(results)}")
    print()

    for i, r in enumerate(results, 1):
        exp_label = (
            "REJECTED"
            if r.score.experience_alignment <= 0.05
            else f"{round(r.score.experience_alignment * 100)}%"
        )
        overall = round(r.score.overall * 100)
        skills_pct = round(r.score.skill_overlap * 100)
        loc_pct = round(r.score.location_match * 100)
        fresh_pct = round(r.score.freshness * 100)

        print(f"  {i:>2}. {r.job.title} @ {r.job.company}")
        print(f"      Overall:{overall}%  Skills:{skills_pct}%"
              f"  Exp:{exp_label}  Loc:{loc_pct}%  Fresh:{fresh_pct}%")
        if r.matched_skills:
            print(f"      Matched : {r.matched_skills[:5]}")
        gaps = [g.missing_skill for g in r.missing_skills[:3]]
        print(f"      Gaps    : {gaps or 'none'}")
        if r.rewrites:
            print(f"      Rewrites: {len(r.rewrites)} bullets")
        print()

    return profile, results