#!/usr/bin/env python3
"""
TEST SCRIPT: Run pipeline with full resume analysis and skill-based job filtering

Shows:
1. Resume skills parsed
2. Job fetch filtered by skills
3. Top 10 jobs with skill match info
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("\n" + "="*70)
print("JOB-RAG PIPELINE TEST: RESUME-DRIVEN JOB MATCHING")
print("="*70 + "\n")

from agents.orchestrator import run_pipeline

try:
    print("="*70)
    print("STEP 1: PARSING RESUME & PREFERENCES")
    print("="*70 + "\n")
    
    profile, results = run_pipeline(
        pdf_path='data/sample_resumes/06749170.pdf',
        experience_level='junior',
        freshness_days=30,
        location_priority=['Hyderabad', 'Bangalore']
    )
    
    print("\n\n" + "="*70)
    print("STEP 2: JOBS FETCHED AND FILTERED BY RESUME SKILLS")
    print("="*70 + "\n")
    
    print("\n" + "="*70)
    print("STEP 3: TOP 10 JOBS MATCHED TO YOUR RESUME")
    print("="*70 + "\n")
    
    if results:
        print(f"✓ Found {len(results)} matching jobs\n")
        
        for i, r in enumerate(results, 1):
            overall = round(r.score.overall * 100)
            skills_pct = round(r.score.skill_overlap * 100)
            exp_label = (
                "REJECTED"
                if r.score.experience_alignment <= 0.05
                else f"{round(r.score.experience_alignment * 100)}%"
            )
            loc_pct = round(r.score.location_match * 100)
            fresh_pct = round(r.score.freshness * 100)
            
            print(f"{i:>2}. {r.job.title}")
            print(f"    Company: {r.job.company}")
            print(f"    Location: {r.job.location}")
            print(f"    ===============================")
            print(f"    Overall Score: {overall}%")
            print(f"    - Skills Match: {skills_pct}%")
            print(f"    - Experience: {exp_label}")
            print(f"    - Location: {loc_pct}%")
            print(f"    - Freshness: {fresh_pct}%")
            
            if r.matched_skills:
                print(f"    ✓ Your skills: {', '.join(r.matched_skills[:5])}")
            
            if r.missing_skills:
                gaps = [g.missing_skill for g in r.missing_skills[:4]]
                print(f"    ✗ Missing skills: {', '.join(gaps)}")
            
            print(f"    Apply: {r.job.apply_url[:60]}...")
            print()
    else:
        print("✗ No matching jobs found")
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
