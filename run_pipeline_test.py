#!/usr/bin/env python3
"""
Quick verification: Run the pipeline with the improved Adzuna fetch
"""
import os
import sys

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("\n" + "="*70)
print("RUNNING PIPELINE WITH IMPROVED ADZUNA FETCH")
print("="*70 + "\n")

from agents.orchestrator import run_pipeline

try:
    profile, results = run_pipeline(
        pdf_path='data/sample_resumes/06749170.pdf',
        experience_level='junior',
        freshness_days=30,
        location_priority=['Hyderabad', 'Bangalore']
    )
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    print(f"Profile: {profile.full_name}")
    print(f"Skills: {len(profile.skills)} found")
    print(f"Results: {len(results)} jobs returned")
    
    if results:
        print(f"\nTop 3 results:")
        for i, r in enumerate(results[:3], 1):
            print(f"  {i}. {r.job.title} @ {r.job.company}")
            print(f"     Overall: {round(r.score.overall*100)}%  Skills: {round(r.score.skill_overlap*100)}%")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
