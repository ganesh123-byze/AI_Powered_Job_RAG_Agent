#!/usr/bin/env python
"""Quick debug script to test Adzuna fetch"""
from agents.orchestrator import run_pipeline

print("\n" + "="*60)
print("RUNNING DEBUG FETCH TEST")
print("="*60 + "\n")

try:
    profile, results = run_pipeline(
        pdf_path='data/sample_resumes/06749170.pdf',
        experience_level='junior',
        freshness_days=30,
        location_priority=['Hyderabad', 'Bangalore']
    )
    print(f'\n\n✓ Final: {len(results)} results')
except Exception as e:
    print(f'\n✗ Error: {e}')
    import traceback
    traceback.print_exc()
