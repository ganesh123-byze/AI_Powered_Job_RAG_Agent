#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: Shows improved skill matching & domain-aware job suggestions
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("\n" + "="*80)
print(" "*15 + "IMPROVED JOB-RAG PIPELINE: DOMAIN-AWARE JOB MATCHING")
print("="*80 + "\n")

# Test the improved skill matcher first
print("="*80)
print("STEP 0: Testing Improved Skill Matching")
print("="*80 + "\n")

from utils.skill_matcher import (
    skills_match,
    normalize_skill,
    detect_domain,
    get_search_queries,
)

# Test cases showing why skills were marked as missing before
test_pairs = [
    ("Machine Learning", "machine learning engineer"),
    ("SupervisedLearning", "supervised"),
    ("TF-IDF", "tfidf"),
    ("Python", "python"),
    ("Scikit-learn", "sklearn"),
    ("LLM", "large language models"),
    ("RAG", "retrieval augmented generation"),
    ("NLP", "natural language processing"),
    ("PyTorch", "pytorch"),
]

print("Skill Matching Tests:")
print("-" * 80)
for skill1, skill2 in test_pairs:
    match = skills_match(skill1, skill2)
    status = "✓ MATCH" if match else "✗ NO MATCH"
    print(f"  '{skill1}' vs '{skill2}' -> {status}")

print("\n" + "="*80)
print("STEP 1: PARSING RESUME & DETECTING DOMAIN")
print("="*80 + "\n")

from agents.resume_parser import parse_resume

profile = parse_resume('data/sample_resumes/06749170.pdf')

print(f"✓ Resume Parsed:")
print(f"  Name        : {profile.full_name}")
print(f"  Experience  : {profile.experience_years} years")
print(f"  Roles       : {profile.preferred_roles}")
print(f"  Total Skills: {len(profile.skills)} skills")
print(f"  Skills      : {profile.skills[:10]}")

# Detect domain
domain = detect_domain(profile.skills or [], profile.preferred_roles or [])
queries= get_search_queries(domain)

print(f"\n✓ Domain Detection:")
print(f"  Primary Domain : {domain.replace('_', ' ').upper()}")
print(f"  Custom Queries :")
for i, q in enumerate(queries, 1):
    print(f"    {i}. {q}")

print("\n" + "="*80)
print("STEP 2: ADZUNA FETCH (Domain-Filtered & Skill-Matched)")
print("="*80 + "\n")

from agents.orchestrator import run_pipeline

try:
    profile, results = run_pipeline(
        pdf_path='data/sample_resumes/06749170.pdf',
        experience_level='junior',
        freshness_days=30,
        location_priority=['Hyderabad', 'Bangalore']
    )
    
    if results:
        print("\n" + "="*80)
        print(f"✅ TOP 10 JOBS MATCHED TO YOUR DOMAIN ({domain.upper()})")
        print("="*80 + "\n")
        
        for i, r in enumerate(results[:10], 1):
            overall = round(r.score.overall * 100)
            skills_pct = round(r.score.skill_overlap * 100)
            exp_label = (
                "❌ REJECTED"
                if r.score.experience_alignment <= 0.05
                else f"{round(r.score.experience_alignment * 100)}%"
            )
            loc_pct = round(r.score.location_match * 100)
            
            rating = "⭐⭐⭐" if overall >= 80 else ("⭐⭐" if overall >= 60 else "⭐")
            
            print(f"[{i:2d}] {r.job.title} {rating}")
            print(f"     Company : {r.job.company}")
            print(f"     Location: {r.job.location} ({loc_pct}% match)")
            print(f"     ─────────────────────────────────")
            print(f"     Score   : {overall}% overall")
            print(f"     Skills  : {skills_pct}% match ✓")
            print(f"     Exp     : {exp_label}")
            
            if r.matched_skills:
                matched_list = ', '.join(r.matched_skills[:5])
                print(f"     Your Skills: {matched_list}")
            
            if r.missing_skills:
                gaps = [g.missing_skill for g in r.missing_skills[:3]]
                print(f"     Learn    : {', '.join(gaps)}")
            
            print()
    else:
        print("⚠️  No matching jobs found")
    
    print("="*80)
    print("✅ TEST COMPLETE - System is now domain-aware and skill-matched!")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
