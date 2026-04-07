#!/usr/bin/env python
"""Direct test of Adzuna API"""
import requests
from config import ADZUNA_APP_ID, ADZUNA_API_KEY, ADZUNA_BASE_URL, ADZUNA_COUNTRY

print(f"ADZUNA_APP_ID: {ADZUNA_APP_ID}")
print(f"ADZUNA_API_KEY: {ADZUNA_API_KEY[:20]}...")
print(f"ADZUNA_BASE_URL: {ADZUNA_BASE_URL}")
print(f"ADZUNA_COUNTRY: {ADZUNA_COUNTRY}\n")

# Test 1: Simple query for "Machine Learning"
url = f"{ADZUNA_BASE_URL}/{ADZUNA_COUNTRY}/search/1"
params = {
    "app_id": ADZUNA_APP_ID,
    "app_key": ADZUNA_API_KEY,
    "results_per_page": 20,
    "what": "Machine Learning Engineer",
    "where": "Bangalore",
}

print("Test 1: Searching for 'Machine Learning Engineer' in Bangalore...")
print(f"URL: {url}")
print(f"Params: what={params['what']}, where={params['where']}\n")

try:
    response = requests.get(url, params=params, timeout=15)
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"Response keys: {data.keys()}")
    print(f"Total: {data.get('count', 'N/A')}")
    print(f"Results count: {len(data.get('results', []))}")
    
    if data.get('results'):
        print(f"\nFirst 3 results:")
        for i, job in enumerate(data.get('results', [])[:3]):
            print(f"  [{i+1}] {job.get('title')} @ {job.get('company', {}).get('display_name')}")
    else:
        print("\n✗ No results returned")
        print(f"Full response: {data}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
