"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF)
Combines: Vector Search + BM25 + Graph Search → RRF Fusion → Reranking

RRF Formula: score(d) = Σ 1/(k + rank_i(d))   where k=60
The k=60 constant was empirically determined to reduce the impact of high outlier ranks.
"""

import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from enum import Enum
from loguru import logger

from app.vector_db.chroma_store import VectorStore
from app.retriever.bm25_retriever import BM25Retriever
from app.graph.neo4j_graph import GraphRetriever


class QueryType(str, Enum):
    DEFINITION = "definition"
    FORMULA = "formula"
    NUMERICAL = "numerical"
    COMPARISON = "comparison"
    CONCEPTUAL = "conceptual"
    MULTI_HOP = "multi_hop"


# Query classification keywords
QUERY_TYPE_SIGNALS = {
    QueryType.DEFINITION: ["what is", "define", "definition", "meaning of", "explain what"],
    QueryType.FORMULA: ["formula", "equation", "expression", "derive", "derivation",
                         "mathematical", "calculate", "compute"],
    QueryType.NUMERICAL: ["find", "calculate", "value of", "how much", "how many",
                           "determine", "evaluate", "given that"],
    QueryType.COMPARISON: ["difference between", "compare", "vs", "versus", "similar",
                            "distinguish", "contrast", "both"],
    QueryType.MULTI_HOP: ["relationship", "how does", "why does", "what happens when",
                           "effect of", "consequence"],
    QueryType.CONCEPTUAL: ["explain", "describe", "how", "why", "what causes",
                            "principle", "concept"],
}


def classify_query(query: str) -> QueryType:
    """Classify query type to route retrieval strategy."""
    query_lower = query.lower()
    scores = {qt: 0 for qt in QueryType}

    for qtype, signals in QUERY_TYPE_SIGNALS.items():
        for signal in signals:
            if signal in query_lower:
                scores[qtype] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return QueryType.CONCEPTUAL
    return best


class HybridRetriever:
    """
    Hybrid retrieval combining three complementary search strategies:
    
    1. Vector Search: Dense retrieval for semantic similarity (paraphrases, synonyms)
    2. BM25 Search: Sparse retrieval for exact keyword matching (formulas, names)
    3. Graph Search: Structural retrieval following concept relationships
    
    RRF fusion is applied to normalize and combine ranked lists from all three.
    This outperforms any single method alone, especially for physics where:
    - "Coulomb's law" needs exact keyword match (BM25 wins)
    - "force between charges" needs semantic match (Vector wins)
    - "concepts related to electrostatics" needs graph traversal (Graph wins)
    """

    RRF_K = 60

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        graph_retriever: Optional[GraphRetriever] = None,
        top_k: int = 10,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.graph_retriever = graph_retriever
        self.top_k = top_k

    def _rrf_fusion(self, ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion across multiple ranked lists.
        
        RRF(d) = Σ_i  1 / (k + rank_i(d))
        
        Benefits:
        - Rank-based: not sensitive to different score scales across retrievers
        - Robust: a document ranked high in ANY list gets boosted
        - Simple: no learned parameters, no tuning needed
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, Dict] = {}

        for ranked_list in ranked_lists:
            for rank, hit in enumerate(ranked_list, start=1):
                cid = hit["chunk_id"]
                rrf_scores[cid] += 1.0 / (k + rank)
                if cid not in chunk_data:
                    chunk_data[cid] = hit

        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for cid, rrf_score in sorted_chunks:
            hit = chunk_data[cid].copy()
            hit["rrf_score"] = round(rrf_score, 6)
            fused.append(hit)

        return fused

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_vector: bool = True,
        use_bm25: bool = True,
        use_graph: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        """
        Main retrieval method. Returns (fused_results, metadata).
        metadata contains per-source results and timing info.
        """
        k = top_k or self.top_k
        query_type = classify_query(query)

        meta = {
            "query_type": query_type.value,
            "vector_results": [],
            "bm25_results": [],
            "graph_results": [],
            "timing": {},
        }

        ranked_lists = []

        # ── Vector Retrieval ──────────────────────────────────────────
        if use_vector:
            t0 = time.perf_counter()
            vector_hits = self.vector_store.search(query, top_k=k)
            meta["timing"]["vector_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            meta["vector_results"] = vector_hits
            if vector_hits:
                ranked_lists.append(vector_hits)

        # ── BM25 Retrieval ────────────────────────────────────────────
        if use_bm25:
            t0 = time.perf_counter()
            bm25_hits = self.bm25_retriever.search(query, top_k=k)
            meta["timing"]["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            meta["bm25_results"] = bm25_hits
            if bm25_hits:
                ranked_lists.append(bm25_hits)

        # ── Graph Retrieval ───────────────────────────────────────────
        if use_graph and self.graph_retriever and self.graph_retriever.available:
            t0 = time.perf_counter()
            graph_hits = self.graph_retriever.search(query, top_k=k)
            meta["timing"]["graph_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            meta["graph_results"] = graph_hits
            if graph_hits:
                ranked_lists.append(graph_hits)

        # ── RRF Fusion ────────────────────────────────────────────────
        if not ranked_lists:
            return [], meta

        t0 = time.perf_counter()
        fused = self._rrf_fusion(ranked_lists, k=self.RRF_K)
        meta["timing"]["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Annotate with per-source scores for explainability
        vector_ids = {h["chunk_id"]: h.get("vector_score", 0) for h in meta["vector_results"]}
        bm25_ids = {h["chunk_id"]: h.get("bm25_score", 0) for h in meta["bm25_results"]}
        graph_ids = {h["chunk_id"]: h.get("graph_score", 0) for h in meta["graph_results"]}

        for hit in fused:
            cid = hit["chunk_id"]
            hit["vector_score"] = vector_ids.get(cid, 0.0)
            hit["bm25_score"] = bm25_ids.get(cid, 0.0)
            hit["graph_score"] = graph_ids.get(cid, 0.0)

        logger.debug(
            f"Retrieval complete: {len(fused)} fused results "
            f"(vec={len(meta['vector_results'])}, "
            f"bm25={len(meta['bm25_results'])}, "
            f"graph={len(meta['graph_results'])})"
        )

        return fused[:k * 2], meta  # return 2x for reranker to select from
