"""
Observability Module — LangSmith tracing + structured logging + analytics dashboard data.

Logs every RAG pipeline step with:
  - Query, query type
  - Retrieval time per source
  - Reranking time
  - Generation time + total latency
  - Number of chunks, citations
  - Answer grounded / not-found flag
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

# LangSmith optional integration
try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_helpers import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(func=None, **kwargs):
        """No-op decorator when langsmith not installed."""
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

LOG_DIR = Path("./data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Structured log file for analytics
QUERY_LOG_FILE = LOG_DIR / "query_logs.jsonl"


class RAGObserver:
    """
    Centralized observability for the RAG pipeline.
    
    Captures:
    - Per-query structured logs (JSONL format)
    - LangSmith traces (when API key configured)
    - Aggregate analytics (latency p50/p95, grounding rate, retrieval hit rates)
    """

    def __init__(self):
        self.langsmith_client: Optional[object] = None
        self._setup_langsmith()
        self._session_logs: List[Dict] = []

    def _setup_langsmith(self):
        if not LANGSMITH_AVAILABLE:
            logger.info("LangSmith not installed — local logging only")
            return
        api_key = os.environ.get("LANGCHAIN_API_KEY")
        if not api_key:
            logger.info("LANGCHAIN_API_KEY not set — LangSmith tracing disabled")
            return
        try:
            self.langsmith_client = LangSmithClient(api_key=api_key)
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "physics-hybrid-rag")
            logger.success("LangSmith tracing enabled")
        except Exception as e:
            logger.warning(f"LangSmith init failed: {e}")

    def log_query(
        self,
        query: str,
        query_type: str,
        answer: str,
        citations: List[Dict],
        chunks: List[Dict],
        latency: Dict,
        retrieval_meta: Dict,
        model: str,
    ) -> Dict:
        """Log a complete RAG query event."""
        is_grounded = "not found in the provided" not in answer.lower()
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "query_type": query_type,
            "model": model,
            "is_grounded": is_grounded,
            "answer_length_words": len(answer.split()),
            "num_chunks": len(chunks),
            "num_citations": len(citations),
            "citation_pages": [c.get("page") for c in citations],
            "latency": latency,
            "retrieval_counts": {
                "vector": len(retrieval_meta.get("vector_results", [])),
                "bm25": len(retrieval_meta.get("bm25_results", [])),
                "graph": len(retrieval_meta.get("graph_results", [])),
            },
            "retrieval_timing_ms": retrieval_meta.get("timing", {}),
            "top_chunk_scores": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "page": c.get("metadata", {}).get("page"),
                    "vector_score": c.get("vector_score", 0),
                    "bm25_score": c.get("bm25_score", 0),
                    "rrf_score": c.get("rrf_score", 0),
                    "reranker_score": c.get("reranker_score"),
                }
                for c in chunks[:3]
            ],
        }

        # Append to JSONL
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Keep in-memory session log (last 200)
        self._session_logs.append(log_entry)
        if len(self._session_logs) > 200:
            self._session_logs = self._session_logs[-200:]

        logger.info(
            f"[RAG] query_type={query_type} grounded={is_grounded} "
            f"chunks={len(chunks)} total_ms={latency.get('total_ms', 0)}"
        )
        return log_entry

    def get_analytics(self) -> Dict:
        """Compute aggregate analytics from session logs."""
        logs = self._session_logs
        if not logs:
            return {"message": "No queries logged yet"}

        total = len(logs)
        grounded = sum(1 for l in logs if l["is_grounded"])
        latencies = [l["latency"].get("total_ms", 0) for l in logs]
        latencies_sorted = sorted(latencies)

        def percentile(data, p):
            if not data:
                return 0
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        type_counts = {}
        for l in logs:
            t = l.get("query_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        retrieval_avg = {
            "vector_ms": round(
                sum(l["retrieval_timing_ms"].get("vector_ms", 0) for l in logs) / total, 1
            ),
            "bm25_ms": round(
                sum(l["retrieval_timing_ms"].get("bm25_ms", 0) for l in logs) / total, 1
            ),
            "graph_ms": round(
                sum(l["retrieval_timing_ms"].get("graph_ms", 0) for l in logs if l["retrieval_timing_ms"].get("graph_ms")) / max(1, sum(1 for l in logs if l["retrieval_timing_ms"].get("graph_ms"))), 1
            ),
        }

        return {
            "total_queries": total,
            "grounding_rate_pct": round(100 * grounded / total, 1),
            "latency": {
                "p50_ms": percentile(latencies_sorted, 50),
                "p90_ms": percentile(latencies_sorted, 90),
                "p95_ms": percentile(latencies_sorted, 95),
                "avg_ms": round(sum(latencies) / total, 1),
            },
            "query_type_distribution": type_counts,
            "retrieval_avg_ms": retrieval_avg,
            "avg_chunks_per_query": round(
                sum(l["num_chunks"] for l in logs) / total, 1
            ),
            "avg_citations_per_query": round(
                sum(l["num_citations"] for l in logs) / total, 1
            ),
        }

    def get_failure_analysis(self) -> Dict:
        """Identify failure patterns from logs."""
        logs = self._session_logs
        if not logs:
            return {"message": "No data yet"}

        not_found = [l for l in logs if not l["is_grounded"]]
        slow_queries = [l for l in logs if l["latency"].get("total_ms", 0) > 5000]
        zero_graph = [l for l in logs if l["retrieval_counts"].get("graph", 0) == 0]

        return {
            "not_found_rate_pct": round(100 * len(not_found) / max(len(logs), 1), 1),
            "not_found_queries": [l["query"] for l in not_found[:5]],
            "slow_query_count": len(slow_queries),
            "slow_queries": [
                {"query": l["query"], "ms": l["latency"].get("total_ms")}
                for l in slow_queries[:5]
            ],
            "graph_miss_rate_pct": round(100 * len(zero_graph) / max(len(logs), 1), 1),
            "recommendations": _generate_recommendations(not_found, slow_queries, zero_graph),
        }


def _generate_recommendations(not_found, slow_queries, zero_graph) -> List[str]:
    recs = []
    if len(not_found) > 0:
        recs.append("Increase TOP_K_RETRIEVAL or expand chunk overlap for better coverage")
    if len(slow_queries) > 0:
        recs.append("Consider quantizing the reranker model or caching frequent queries")
    if len(zero_graph) > 0:
        recs.append("Verify Neo4j is running and the knowledge graph is built")
    if not recs:
        recs.append("System performing well — no critical issues detected")
    return recs


# Singleton observer
observer = RAGObserver()
