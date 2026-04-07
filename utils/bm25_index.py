"""
bm25_index.py  –  BM25 Index Manager
======================================
Improvements:
  1. _preprocess_text() preserves tech tokens (C++, C#, .NET, node.js)
  2. build_from_chroma() indexes matched_skills from metadata (big BM25 boost)
  3. search() accepts a skill list query for targeted lookup
  4. Index is cleared and rebuilt on every pipeline run (no stale data)
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from rank_bm25 import BM25Okapi


class BM25IndexManager:

    def __init__(self, index_dir: str = "data/bm25_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_file    = self.index_dir / "bm25_index.pkl"
        self.metadata_file = self.index_dir / "job_metadata.json"
        self.corpus_file   = self.index_dir / "corpus.json"

        self.bm25: Optional[BM25Okapi] = None
        self.job_metadata: Dict[str, Dict] = {}
        self.corpus: List[List[str]] = []

        self._load_index()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self):
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    self.bm25 = pickle.load(f)
                with open(self.metadata_file, "r") as f:
                    self.job_metadata = json.load(f)
                with open(self.corpus_file, "r") as f:
                    self.corpus = json.load(f)
                print(f"  BM25: loaded {len(self.job_metadata)} jobs from disk")
            except Exception as exc:
                print(f"  BM25: load failed ({exc}), will rebuild")
                self._reset()

    def _save_index(self):
        try:
            with open(self.index_file, "wb") as f:
                pickle.dump(self.bm25, f)
            with open(self.metadata_file, "w") as f:
                json.dump(self.job_metadata, f)
            with open(self.corpus_file, "w") as f:
                json.dump(self.corpus, f)
            print(f"  BM25: saved {len(self.job_metadata)} jobs")
        except Exception as exc:
            print(f"  BM25: save failed ({exc})")

    def _reset(self):
        self.bm25 = None
        self.job_metadata = {}
        self.corpus = []

    # ------------------------------------------------------------------
    # Tokenisation  –  preserves tech tokens
    # ------------------------------------------------------------------

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Tokenise text for BM25.
        Preserves important tech tokens: c++, c#, .net, node.js, etc.
        """
        import re

        text = text.lower()

        # Normalise special tech tokens before stripping punctuation
        tech_replacements = {
            "c++": "cplusplus",
            "c#": "csharp",
            ".net": "dotnet",
            "node.js": "nodejs",
            "react.js": "reactjs",
            "next.js": "nextjs",
            "vue.js": "vuejs",
            "scikit-learn": "scikitlearn",
            "sci-kit learn": "scikitlearn",
            "tf-idf": "tfidf",
            "t5": "t5model",
            "ci/cd": "cicd",
            "a/b": "abtesting",
        }
        for orig, repl in tech_replacements.items():
            text = text.replace(orig, repl)

        # Remove URLs and emails
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        # Tokenise: alphanumeric + preserved symbols (+ # .)
        tokens = re.findall(r"\b[a-z0-9][a-z0-9\.\+\#]*\b", text)

        # Remove single-char tokens and pure numbers < 2
        tokens = [t for t in tokens if len(t) >= 2 and not (t.isdigit() and len(t) < 2)]

        return tokens

    # ------------------------------------------------------------------
    # Build from ChromaDB
    # ------------------------------------------------------------------

    def build_from_chroma(self, chroma_results: Dict):
        """Build BM25 index from ChromaDB .get() results."""
        documents = chroma_results.get("documents") or []
        metadatas = chroma_results.get("metadatas") or []

        if not documents:
            print("  BM25: no documents to index")
            return

        self.corpus = []
        self.job_metadata = {}

        for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
            meta = meta or {}

            # Build rich text: title + company + location + matched_skills + desc
            matched_skills = meta.get("matched_skills", "")
            combined = " ".join([
                meta.get("title", ""),
                meta.get("company", ""),
                meta.get("location", ""),
                matched_skills,             # skills stored at fetch time
                matched_skills,             # repeat for higher BM25 weight
                matched_skills,             # repeat again
                (doc or "")[:2500],
            ])

            tokens = self._preprocess_text(combined)
            self.corpus.append(tokens)

            self.job_metadata[str(idx)] = {
                "job_id":          meta.get("job_id", ""),
                "title":           meta.get("title", ""),
                "company":         meta.get("company", ""),
                "location":        meta.get("location", ""),
                "posted_date":     meta.get("posted_date", ""),
                "apply_url":       meta.get("apply_url", ""),
                "salary_min":      meta.get("salary_min"),
                "salary_max":      meta.get("salary_max"),
                "freshness_bucket": meta.get("freshness_bucket", "older"),
                "matched_skills":  matched_skills,
            }

        self.bm25 = BM25Okapi(self.corpus)
        print(f"  BM25: indexed {len(self.corpus)} documents")
        self._save_index()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 50,
        min_score: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """BM25 keyword search. Returns [(job_index, score)]."""
        if self.bm25 is None:
            return []

        tokens = self._preprocess_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        results = [
            (idx, float(score))
            for idx, score in enumerate(scores)
            if score >= min_score
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_skills(
        self,
        skills: List[str],
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        Search using a skill list as query.
        Repeats each skill to give it higher term frequency.
        """
        if not skills:
            return []
        # Repeat each skill 3× so BM25 weighs them highly
        query = " ".join(s for s in skills[:20] for _ in range(3))
        return self.search(query, top_k=top_k)

    def get_job_metadata(self, job_index: int) -> Optional[Dict]:
        return self.job_metadata.get(str(job_index))

    def clear_index(self):
        self._reset()
        for f in [self.index_file, self.metadata_file, self.corpus_file]:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        print("  BM25: index cleared")