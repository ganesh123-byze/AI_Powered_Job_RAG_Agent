"""
embedder.py  –  OpenRouter Embedder
=====================================
Improvements:
  1. build_resume_query_text() repeats top skills 3x (boosts BM25 + cosine)
  2. Adds project tech stack to query for better semantic match
  3. Embedding cache uses text hash (not truncated hash)
  4. Batch embedding with automatic chunking for large lists
"""

from typing import List, Optional
import httpx
import time
import hashlib

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
)


class OpenRouterEmbedder:
    """Generates embeddings using OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        raw_model = model or OPENROUTER_EMBEDDING_MODEL or "text-embedding-3-small"
        self.model = raw_model.replace("openai/", "")
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, text: str) -> List[float]:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        emb = self._call_api([text])[0]
        self._cache[key] = emb
        return emb

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with automatic chunking (max 20 per request)."""
        all_embeddings: list[list[float]] = []
        chunk_size = 20
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            all_embeddings.extend(self._call_api(chunk))
            if i + chunk_size < len(texts):
                time.sleep(0.2)
        return all_embeddings

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **OPENROUTER_HEADERS,
        }
        payload = {"model": self.model, "input": texts}

        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/embeddings",
                        json=payload,
                        headers=headers,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    if "data" not in data:
                        raise RuntimeError(f"Invalid embedding response: {data}")
                    return [item["embedding"] for item in data["data"]]

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"⚠ Embedder rate limited. Waiting {wait}s…")
                    time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    raise ValueError("Invalid OpenRouter API key")
                if resp.status_code == 402:
                    raise ValueError("Insufficient OpenRouter credits")

                raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.json()}")

            except httpx.TimeoutException:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Embedding timeout after {self.timeout}s")
            except (ValueError, RuntimeError):
                raise
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Embedding failed: {exc}") from exc

        raise RuntimeError("Embedding: max retries exceeded")

    def clear_cache(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_embedder: Optional[OpenRouterEmbedder] = None

def get_embedder() -> OpenRouterEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OpenRouterEmbedder()
    return _embedder

def embed_for_retrieval(text: str) -> List[float]:
    return get_embedder().embed(text)


# ---------------------------------------------------------------------------
# Resume query builder  –  IMPROVED
# ---------------------------------------------------------------------------

def build_resume_query_text(profile) -> str:
    """
    Build a rich search query from ResumeProfile.

    Key trick: repeat top skills 3× so BM25 and cosine similarity
    both weight them heavily (same as query expansion).
    """
    parts: list[str] = []

    # 1. Preferred roles (high signal)
    if profile.preferred_roles:
        roles = " ".join(profile.preferred_roles[:4])
        parts.append(f"Roles: {roles}")
        parts.append(roles)   # repeat once

    # 2. Skills – repeat top 10 skills 3× for emphasis
    if profile.skills:
        top = profile.skills[:15]
        skills_str = " ".join(top)
        parts.append(f"Skills: {skills_str}")
        parts.append(skills_str)           # 2nd repeat
        parts.append(" ".join(top[:8]))    # 3rd repeat (top 8 only)

    # 3. Project technologies  (catches frameworks not in skills list)
    if profile.projects:
        all_techs: list[str] = []
        for proj in profile.projects:
            all_techs.extend(proj.technologies[:4])
        if all_techs:
            tech_str = " ".join(set(all_techs))
            parts.append(f"Technologies: {tech_str}")

    # 4. Experience signal
    if profile.experience_years:
        parts.append(f"{profile.experience_years} years experience")

    # 5. Summary (semantic context)
    if profile.summary:
        parts.append(profile.summary[:200])

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Compat wrapper (used by hybrid_retriever.py)
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self):
        self._e = get_embedder()

    def embed(self, text: str) -> List[float]:
        return self._e.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._e.embed_batch(texts)