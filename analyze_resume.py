#!/usr/bin/env python3
"""
RESUME ANALYSIS: Show what the system extracts from your resume
and how it will be used for job matching
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("\n" + "="*80)
print(" "*20 + "RESUME ANALYSIS & JOB MATCHING STRATEGY")
print("="*80 + "\n")

from agents.resume_parser import parse_resume

# Parse the resume
try:
    profile = parse_resume('data/sample_resumes/06749170.pdf')
    
    print("📋 RESUME PARSED SUCCESSFULLY\n")
    print("─" * 80)
    print("CANDIDATE INFORMATION")
    print("─" * 80)
    print(f"Name              : {profile.full_name}")
    print(f"Experience        : {profile.experience_years} years")
    print(f"Education         : {profile.education}")
    print(f"Current Role      : {profile.current_role}")
    
    print(f"\nPreferred Roles   : {profile.preferred_roles}")
    print(f"Summary           : {profile.summary[:150]}..." if profile.summary else "No summary")
    
    print("\n" + "─" * 80)
    print("TECHNICAL SKILLS EXTRACTED")
    print("─" * 80)
    print(f"Total Skills Found: {len(profile.skills)}\n")
    
    # Group skills by category for better visualization
    skill_categories = {
        'Core Programming': ['Python', 'SQL'],
        'Machine Learning': [s for s in profile.skills if any(x in s.lower() for x in ['supervised', 'unsupervised', 'regression', 'classification', 'feature', 'model', 'scikit'])],
        'Statistics & Math': [s for s in profile.skills if any(x in s.lower() for x in ['probability', 'hypothesis', 'statistical', 'distribution'])],
        'NLP & LLMs': [s for s in profile.skills if any(x in s.lower() for x in ['nlp', 'nlp', 'tfidf', 'hugging', 'lm', 'langchain', 'openai', 'rag', 'prompt', 'ollama', 'transformers', 'logistic'])],
        'Deployment & MLOps': [s for s in profile.skills if any(x in s.lower() for x in ['docker', 'rest', 'flask', 'fastapi', 'streamlit', 'api'])],
        'Data & Visualization': [s for s in profile.skills if any(x in s.lower() for x in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'eda', 'power bi', 'jupyter'])],
        'Tools & Databases': [s for s in profile.skills if any(x in s.lower() for x in ['mysql', 'git', 'vscode', 'colab', 'jupyter', 'notebook'])],
    }
    
    for category, skills in skill_categories.items():
        if skills:
            unique_skills = list(dict.fromkeys(skills))  # Dedupe
            print(f"  {category:.<25} {', '.join(unique_skills)}")
    
    print(f"\n  Raw Skills List (All {len(profile.skills)} skills):")
    print(f"  {profile.skills}\n")
    
    print("─" * 80)
    print("💡 JOB SEARCH STRATEGY (What the system will do)")
    print("─" * 80)
    print("""
1. PRIMARY SEARCH QUERIES (in order):
   • "Machine Learning Engineer, Python" (from preferred_roles + top skills)
   • "Machine Learning Engineer" (fallback - just role)
   • "Python" (basic fallback)
   • "Engineer" (ultimate fallback)

2. LOCATIONS TO SEARCH:
   • Hyderabad & Telangana
   • Bangalore & Karnataka
   • (You specified in preferences)

3. JOB FILTERING CRITERIA:
   ✓ Only Junior/Entry-level jobs (per experience_level=junior)
   ✓ Only jobs posted in last 30 days (per freshness_days=30)
   ✓ Only jobs matching your Python/ML skills (NOT security/compliance jobs)

4. RANKING CRITERIA (Top 10):
   • Skill overlap score (highest weight)
   • Experience alignment (junior roles preferred)
   • Location match (Bangalore/Hyderabad)
   • Job freshness (recent posts preferred)
""")
    
    print("─" * 80)
    print("🎯 EXPECTED JOB TITLES (based on your profile)")
    print("─" * 80)
    print("""
   EXPECTED MATCHES:
   ✓ Machine Learning Engineer (Junior/Entry)
   ✓ AI/ML Developer
   ✓ Python Developer (ML focused)
   ✓ Data Scientist (Junior)
   ✓ Data Analyst (Python/ML)
   ✓ Backend Developer (Python)

   UNEXPECTED (will be filtered out):
   ✗ Security Engineer
   ✗ DevOps Engineer
   ✗ Frontend Developer
   ✗ Graphic Designer
   ✗ Cybersecurity Analyst
""")
    
    print("\n" + "="*80)
    print("Now running the FULL PIPELINE to fetch and rank jobs...\n")
    print("="*80 + "\n")
    
    # Now run the full pipeline
    from agents.orchestrator import run_pipeline
    
    profile, results = run_pipeline(
        pdf_path='data/sample_resumes/06749170.pdf',
        experience_level='junior',
        freshness_days=30,
        location_priority=['Hyderabad', 'Bangalore']
    )
    
    print("\n" + "="*80)
    print("✅ TOP 10 JOBS MATCHED TO YOUR RESUME")
    print("="*80 + "\n")
    
    if results:
        for i, r in enumerate(results[:10], 1):
            overall = round(r.score.overall * 100)
            skills_match = round(r.score.skill_overlap * 100)
            exp_label = (
                "❌ REJECTED"
                if r.score.experience_alignment <= 0.05
                else f"{round(r.score.experience_alignment * 100)}%"
            )
            loc_match = round(r.score.location_match * 100)
            fresh_score = round(r.score.freshness * 100)
            
            print(f"[{i:2d}] {r.job.title}")
            print(f"     Company    : {r.job.company}")
            print(f"     Location   : {r.job.location}")
            print(f"     Posted     : {r.job.posted_date} ({fresh_score}% fresh)")
            print(f"     ─────────════════════════")
            print(f"     OVERALL    : {overall}% ⭐{'⭐' if overall >= 70 else ''}")
            print(f"     Your Skills: {skills_match}% ✓")
            print(f"     Experience: {exp_label}")
            print(f"     Location   : {loc_match}% match")
            
            if r.matched_skills:
                matched = ', '.join(r.matched_skills[:6])
                print(f"     ✓ Matched  : {matched}")
            
            if r.missing_skills:
                gaps = [g.missing_skill for g in r.missing_skills[:3]]
                print(f"     ✗ To Learn : {', '.join(gaps)}")
            
            print(f"     Apply      : {r.job.apply_url[:70]}...")
            print()
    else:
        print("⚠️  No matching jobs found")
    
    print("="*80)
    print("\n")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
