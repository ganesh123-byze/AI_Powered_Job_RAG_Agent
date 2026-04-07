"""
hybrid_retriever.py  –  Hybrid BM25 + Vector Retriever
========================================================
Improvements:
  1. BM25 search now also runs a skill-specific query (separate pass)
  2. Vector results are mapped to BM25 indices more robustly
  3. RRF and weighted fusion both available
  4. Results include all metadata fields needed by scorer
  5. Handles empty ChromaDB gracefully
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HybridSearchResult:
    job_index:       int
    job_id:          str
    title:           str
    company:         str
    location:        str
    description:     str
    posted_date:     str
    apply_url:       str
    salary_min:      Optional[float]
    salary_max:      Optional[float]
    freshness_bucket: str
    bm25_score:      float
    vector_score:    float
    hybrid_score:    float


class HybridRetriever:

    def __init__(
        self,
        bm25_manager,
        chroma_client,
        embedder,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25_manager
        self.chroma = chroma_client
        self.embedder = embedder

        total = bm25_weight + vector_weight
        self.bm25_w   = bm25_weight  / total
        self.vector_w = vector_weight / total
        self.rrf_k    = rrf_k

    # ------------------------------------------------------------------
    # Normalisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(scores: List[float]) -> List[float]:
        if not scores:
            return []
        arr = np.array(scores, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return [0.5] * len(scores)
        return ((arr - lo) / (hi - lo)).tolist()

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    def _rrf(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_results, 1):
            scores[idx] = scores.get(idx, 0.0) + 1 / (self.rrf_k + rank)
        for rank, (idx, _) in enumerate(vector_results, 1):
            scores[idx] = scores.get(idx, 0.0) + 1 / (self.rrf_k + rank)
        return scores

    def _weighted(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        bm25_norm   = self._norm([s for _, s in bm25_results])
        vector_norm = self._norm([s for _, s in vector_results])

        bm25_map   = {idx: bm25_norm[i]   for i, (idx, _) in enumerate(bm25_results)}
        vector_map = {idx: vector_norm[i] for i, (idx, _) in enumerate(vector_results)}

        all_idx = set(bm25_map) | set(vector_map)
        return {
            idx: bm25_map.get(idx, 0.0)   * self.bm25_w
               + vector_map.get(idx, 0.0) * self.vector_w
            for idx in all_idx
        }

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 50,
        use_rrf: bool = True,
        debug: bool = False,
        skill_list: Optional[List[str]] = None,
    ) -> List[HybridSearchResult]:
        """
        Retrieve jobs using hybrid BM25 + vector search.

        Args:
            query_text: Natural language query string
            query_embedding: Pre-computed embedding vector
            top_k: Number of final results
            use_rrf: True = RRF fusion, False = weighted fusion
            debug: Verbose output
            skill_list: Optional skill list for a secondary BM25 pass
        """

        # ── BM25 pass 1: full query ──────────────────────────────────────
        bm25_results = self.bm25.search(query_text, top_k=top_k)

        # ── BM25 pass 2: skill-specific query (extra signal) ────────────
        if skill_list:
            skill_results = self.bm25.search_by_skills(skill_list, top_k=top_k)
            # Merge: take best score per index from both passes
            bm25_map_merged: dict[int, float] = dict(bm25_results)
            for idx, score in skill_results:
                bm25_map_merged[idx] = max(bm25_map_merged.get(idx, 0.0), score)
            bm25_results = sorted(bm25_map_merged.items(), key=lambda x: x[1], reverse=True)[:top_k]

        if debug:
            print(f"  [BM25] {len(bm25_results)} results")
            for rank, (idx, score) in enumerate(bm25_results[:5], 1):
                m = self.bm25.get_job_metadata(idx) or {}
                print(f"    {rank}. {m.get('title')} (score={score:.3f})")

        # ── Vector search via ChromaDB ────────────────────────────────────
        try:
            vr = self.chroma.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.chroma.count() or 1),
                include=["distances", "documents", "metadatas"],
            )
        except Exception:
            vr = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        # Fallback when collection returns empty nested lists
        if not vr or vr.get("ids") == [[]]:
            try:
                from utils.chroma_client import query_jobs
                vr = query_jobs(query_embedding, n_results=top_k)
            except Exception:
                vr = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        vector_ids       = vr.get("ids", [[]])[0]
        vector_distances = vr.get("distances", [[]])[0]

        # Cosine distance → similarity (ChromaDB cosine: 0=identical, 2=opposite)
        vector_sims = [max(0.0, 1.0 - d) for d in vector_distances]

        # Map Chroma job_id → BM25 numeric index
        jobid_to_idx: dict[str, int] = {}
        for k, meta in self.bm25.job_metadata.items():
            try:
                bidx = int(k)
            except Exception:
                continue
            jid = meta.get("job_id", "")
            if jid:
                jobid_to_idx[str(jid)] = bidx

        vector_results: List[Tuple[int, float]] = []
        for jid, sim in zip(vector_ids, vector_sims):
            jid_str = str(jid)
            mapped  = jobid_to_idx.get(jid_str)
            if mapped is None:
                # Try direct integer index
                try:
                    mapped = int(jid_str)
                except Exception:
                    continue
            vector_results.append((mapped, sim))

        if debug:
            print(f"  [Vector] {len(vector_results)} results")

        # ── Fusion ───────────────────────────────────────────────────────
        fused = (
            self._rrf(bm25_results, vector_results)
            if use_rrf
            else self._weighted(bm25_results, vector_results)
        )

        # ── Build result objects ──────────────────────────────────────────
        bm25_map   = dict(bm25_results)
        vector_map = dict(vector_results)

        results: List[HybridSearchResult] = []
        for idx, hybrid_score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            meta = self.bm25.get_job_metadata(idx)
            if not meta:
                continue

            def _safe_float(v) -> Optional[float]:
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            results.append(HybridSearchResult(
                job_index=idx,
                job_id=meta.get("job_id", ""),
                title=meta.get("title", ""),
                company=meta.get("company", ""),
                location=meta.get("location", ""),
                description="",
                posted_date=meta.get("posted_date", ""),
                apply_url=meta.get("apply_url", ""),
                salary_min=_safe_float(meta.get("salary_min")),
                salary_max=_safe_float(meta.get("salary_max")),
                freshness_bucket=meta.get("freshness_bucket", "older"),
                bm25_score=round(bm25_map.get(idx, 0.0), 4),
                vector_score=round(vector_map.get(idx, 0.0), 4),
                hybrid_score=round(hybrid_score, 4),
            ))

        if debug:
            print(f"  [Hybrid] {len(results)} fused results (top 5):")
            for r in results[:5]:
                print(f"    {r.title} | BM25:{r.bm25_score:.4f} "
                      f"Vec:{r.vector_score:.4f} Hybrid:{r.hybrid_score:.4f}")

        return results