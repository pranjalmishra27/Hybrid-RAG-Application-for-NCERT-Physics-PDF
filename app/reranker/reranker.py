"""
Reranker Module using BAAI/bge-reranker-base
Cross-encoder reranking: takes (query, passage) pair and scores relevance.

Why reranking?
- First-stage retrieval (vector + BM25) uses bi-encoder: fast but approximate
- Reranker uses cross-encoder: slower but much more accurate
- Bi-encoder: encodes query and doc independently → fast retrieval
- Cross-encoder: attends query+doc together → better relevance judgment
- Typical pipeline: retrieve top-20 → rerank → return top-5
"""

import time
from typing import List, Dict, Optional
from loguru import logger

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class Reranker:
    """BGE Cross-Encoder reranker for final passage selection."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model: Optional[object] = None

    @property
    def model(self):
        if self._model is None and RERANKER_AVAILABLE:
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank candidate chunks using cross-encoder.
        
        Args:
            query: User query
            candidates: List of retrieved chunks (from hybrid retriever)
            top_k: Number of final chunks to return
            
        Returns:
            Top-k reranked chunks with reranker_score added
        """
        if not candidates:
            return []

        # Limit input to prevent OOM
        candidates = candidates[:20]

        if self.model is None:
            logger.warning("Reranker not available. Using RRF scores as fallback.")
            return sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)[:top_k]

        t0 = time.perf_counter()

        # Create query-passage pairs for cross-encoder
        pairs = [(query, c["content"][:512]) for c in candidates]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            elapsed = (time.perf_counter() - t0) * 1000

            # Attach scores and sort
            for i, chunk in enumerate(candidates):
                chunk["reranker_score"] = round(float(scores[i]), 4)

            reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)

            logger.debug(
                f"Reranked {len(candidates)} → top {top_k} "
                f"in {elapsed:.1f}ms. "
                f"Best score: {reranked[0]['reranker_score']:.4f}"
            )

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to RRF order.")
            return sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)[:top_k]

    def is_available(self) -> bool:
        return RERANKER_AVAILABLE and self.model is not None
