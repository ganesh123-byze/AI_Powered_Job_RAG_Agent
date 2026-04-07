"""
Test script to debug Adzuna fetch with improved what_query
Run this with: python test_adzuna_fetch_debug.py
"""

import sys
sys.path.insert(0, '/d/job-rag-agent')

import requests
from config import ADZUNA_APP_ID, ADZUNA_API_KEY, ADZUNA_BASE_URL, ADZUNA_COUNTRY

print("\n" + "="*70)
print("ADZUNA API TEST - Testing improved what_query")
print("="*70)

# Test what_query values
test_queries = [
    "Machine Learning Engineer, Python",
    "AI Engineer, Python",
    "Python Developer",
    "Machine Learning",
]

for what_query in test_queries:
    print(f"\n--- Testing: '{what_query}' in Bangalore ---")
    
    url = f"{ADZUNA_BASE_URL}/{ADZUNA_COUNTRY}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "results_per_page": 5,
        "what": what_query,
        "where": "Bangalore",
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            count = len(data.get('results', []))
            total = data.get('count', 'N/A')
            print(f"✓ Found {count} results (Total: {total})")
            
            if data.get('results'):
                print("  First result:")
                job = data['results'][0]
                print(f"    Title: {job.get('title')}")
                print(f"    Company: {job.get('company',{}).get('display_name')}")
                print(f"    Location: {job.get('location',{}).get('display_name')}")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"✗ Exception: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
