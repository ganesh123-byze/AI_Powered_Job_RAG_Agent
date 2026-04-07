import os
from dotenv import load_dotenv

load_dotenv()

# ===== OPENROUTER API CONFIGURATION =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_LLM_MODEL = "google/gemini-2.0-flash-001"
OPENROUTER_EMBEDDING_MODEL = "text-embedding-3-small"
OPENROUTER_TIMEOUT = 60

OPENROUTER_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
    "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Agentic RAG Job Matcher"),
}

# Validate API key on import
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file!")

# ===== ADZUNA API (for job fetching) =====
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
ADZUNA_COUNTRY = "in"
ADZUNA_RESULTS_PER_PAGE = 20
ADZUNA_MAX_DAYS_OLD = 30

# ===== CHROMADB =====
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_RESUME = "resume_profiles"
COLLECTION_JOBS = "job_listings"

# ===== BM25 INDEX =====
BM25_INDEX_PATH = "data/bm25_index"

# ===== SCORING WEIGHTS =====
SCORE_WEIGHTS = {
    "skill_overlap": 0.40,
    "experience_alignment": 0.25,
    "location_match": 0.20,
    "freshness": 0.15,
}

# ===== RETRIEVAL =====
TOP_K_RETRIEVE = 50
TOP_K_RETURN = 10

# ===== HYBRID SEARCH =====
HYBRID_SEARCH = {
    "enabled": True,
    "bm25_weight": 0.4,
    "vector_weight": 0.6,
    "rrf_k": 60,
    "min_bm25_score": 0.0,
    "use_rrf": True,
    "debug": False,
}

# ===== FRESHNESS SCORES =====
FRESHNESS_SCORES = {
    "today": 1.0,
    "this_week": 0.75,
    "this_month": 0.40,
    "older": 0.10,
}

# ===== PERFORMANCE TUNING =====
MAX_CONCURRENT_OPENROUTER_CALLS = 3
CACHE_EMBEDDINGS = True
ENABLE_WEB_SEARCH_FALLBACK = True
ENABLE_RESUME_REWRITING = True
ENABLE_GAP_ANALYSIS = True
