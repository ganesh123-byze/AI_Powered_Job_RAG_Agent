import json
import hashlib
from pathlib import Path
from models.resume import ResumeProfile

CACHE_DIR = Path("data/resume_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_key(pdf_path: str) -> str:
    """Get unique cache key for PDF."""
    return hashlib.md5(pdf_path.encode()).hexdigest()


def get_cached_profile(pdf_path: str) -> ResumeProfile | None:
    """Get cached resume profile if exists."""
    cache_key = _get_cache_key(pdf_path)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        return ResumeProfile(**data)
    except Exception:
        return None


def save_profile_to_cache(pdf_path: str, profile: ResumeProfile):
    """Save resume profile to cache."""
    cache_key = _get_cache_key(pdf_path)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    try:
        with open(cache_file, "w") as f:
            json.dump(profile.model_dump(), f, indent=2)
    except Exception as e:
        print(f"  Cache save failed: {e}")


def clear_cache():
    """Clear all cached profiles."""
    for file in CACHE_DIR.glob("*.json"):
        file.unlink()
    print(f"  Cache cleared ({CACHE_DIR})")
