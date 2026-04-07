#!/usr/bin/env python3
"""
TEST: Validate experience level filtering works correctly
Shows which jobs are kept/rejected based on experience level
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("\n" + "="*80)
print(" "*15 + "TEST: EXPERIENCE LEVEL FILTERING")
print("="*80 + "\n")

from agents.job_fetcher import _is_suitable_for_level
from models.job import JobListing

# Test cases: (title, description, experience_level, should_be_kept)
test_cases = [
    # FRESHER LEVEL
    (
        "Fresher Python Developer",
        "Entry level position. No prior experience required. 0-1 year preferred.",
        "fresher",
        True,  # ✓ Should keep
    ),
    (
        "Senior Python Developer",
        "5+ years of experience required. Senior level position.",
        "fresher",
        False,  # ✗ Should reject
    ),
    (
        "Junior ML Engineer",
        "1-2 years of experience. Entry to junior level.",
        "fresher",
        False,  # ✗ Should reject (junior, not fresher)
    ),
    
    # JUNIOR LEVEL
    (
        "Junior Data Scientist",
        "1-2 years of experience required. Perfect for someone starting their career.",
        "junior",
        True,  # ✓ Should keep
    ),
    (
        "Senior Data Scientist",
        "5+ years of experience required. Senior level position.",
        "junior",
        False,  # ✗ Should reject
    ),
    (
        "Python Developer",
        "3-5 years of experience. Mid-level position.",
        "junior",
        False,  # ✗ Should reject (mid-level, not junior)
    ),
    (
        "Machine Learning Engineer",
        "7 years of experience minimum. Senior technical stack.",
        "junior",
        False,  # ✗ Should reject
    ),
    
    # MID LEVEL
    (
        "Mid-level Engineer",
        "3-5 years of experience required. Intermediate level.",
        "mid",
        True,  # ✓ Should keep
    ),
    (
        "Principal Engineer",
        "10+ years of experience. Leadership experience required.",
        "mid",
        False,  # ✗ Should reject
    ),
    (
        "Associate Developer",
        "1-2 years experience. Junior level.",
        "mid",
        False,  # ✗ Should reject (too junior)
    ),
]

print("Testing Experience Level Filtering:\n")
print("-" * 80)

passed = 0
failed = 0

for title, description, level, should_keep in test_cases:
    job = JobListing(
        job_id="test",
        title=title,
        company="Test Co",
        location="Test",
        description=description,
        posted_date="2026-04-06",
        apply_url="http://test",
        salary_min=None,
        salary_max=None,
        freshness_bucket="1_week",
    )
    
    is_suitable = _is_suitable_for_level(job, level)
    expected = "KEEP ✓" if should_keep else "REJECT ✗"
    actual = "KEEP ✓" if is_suitable else "REJECT ✗"
    correct = (is_suitable == should_keep)
    
    status = "✓ PASS" if correct else "✗ FAIL"
    if correct:
        passed += 1
    else:
        failed += 1
    
    print(f"{status}: [{level.upper():6s}] {title}")
    print(f"       Expected: {expected}  |  Actual: {actual}")
    if not correct:
        print(f"       ⚠️  MISMATCH!")
    print()

print("-" * 80)
print(f"\nResults: {passed} Passed, {failed} Failed")

if failed == 0:
    print("✅ All tests passed! Experience level filtering is working correctly.")
else:
    print(f"❌ {failed} test(s) failed. Check the filtering logic.")

print("\n" + "="*80)
print("Now running full pipeline test...\n")
print("="*80 + "\n")

from agents.orchestrator import run_pipeline

try:
    profile, results = run_pipeline(
        pdf_path='data/sample_resumes/06749170.pdf',
        experience_level='junior',  # JUNIOR level
        freshness_days=30,
        location_priority=['Hyderabad', 'Bangalore']
    )
    
    print("\n" + "="*80)
    print("✅ FULL PIPELINE RESULTS (Junior Level)")
    print("="*80 + "\n")
    
    if results:
        for i, r in enumerate(results[:5], 1):
            exp_score = round(r.score.experience_alignment * 100)
            print(f"[{i}] {r.job.title} @ {r.job.company}")
            print(f"    Experience Score: {exp_score}%")
            
            # Check if any have been rejected
            if exp_score <= 5:
                print(f"     ⚠️  REJECTED (experience level)")
            print()
    else:
        print("No jobs found")
    
    print("="*80)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
