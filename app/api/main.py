"""
FastAPI Backend for Physics Hybrid RAG
All retrieval, generation, and management endpoints.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Lazy-loaded components ────────────────────────────────────────────────────
from app.ingestion.pdf_parser import PDFIngestionPipeline
from app.vector_db.chroma_store import VectorStore
from app.retriever.bm25_retriever import BM25Retriever
from app.graph.neo4j_graph import GraphRetriever
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.query_understanding import classify_query, get_query_summary
from app.reranker.reranker import Reranker
from app.api.llm_client import LLMClient
from app.api.observability import observer

# ── Global state ──────────────────────────────────────────────────────────────
_vector_store: Optional[VectorStore] = None
_bm25_retriever: Optional[BM25Retriever] = None
_graph_retriever: Optional[GraphRetriever] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_reranker: Optional[Reranker] = None
_llm_client: Optional[LLMClient] = None
_conversation_history: List[Dict] = []
_ingestion_status = {"status": "not_started", "message": "", "chunks": 0}


def get_components():
    global _vector_store, _bm25_retriever, _graph_retriever
    global _hybrid_retriever, _reranker, _llm_client

    if _vector_store is None:
        _vector_store = VectorStore(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        )
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever(
            index_path=os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
        )
        _bm25_retriever.load()

    if _graph_retriever is None:
        _graph_retriever = GraphRetriever(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "physics_rag_2024"),
        )
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(
            vector_store=_vector_store,
            bm25_retriever=_bm25_retriever,
            graph_retriever=_graph_retriever,
            top_k=int(os.getenv("TOP_K_RETRIEVAL", "10")),
        )
    if _reranker is None:
        _reranker = Reranker(model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))

    if _llm_client is None:
        _llm_client = LLMClient(provider=os.getenv("LLM_PROVIDER", "gemini"))

    return _vector_store, _bm25_retriever, _graph_retriever, _hybrid_retriever, _reranker, _llm_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Physics RAG API starting up...")
    # Pre-initialize components
    try:
        get_components()
        logger.success("All components initialized")
    except Exception as e:
        logger.warning(f"Component init warning: {e}")
    yield
    logger.info("Shutting down Physics RAG API")


app = FastAPI(
    title="Physics Hybrid RAG API",
    description="Production-grade Hybrid RAG for NCERT Class 12 Physics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    use_vector: bool = True
    use_bm25: bool = True
    use_graph: bool = True
    use_reranker: bool = True
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict]
    chunks: List[Dict]
    query_type: str
    retrieval_meta: Dict
    latency: Dict
    model: str


class IngestRequest(BaseModel):
    pdf_path: str = "./data/ncert_physics_part1.pdf"
    chunk_size: int = 900
    chunk_overlap: int = 175
    rebuild_graph: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Physics Hybrid RAG",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/query", "/ingest", "/status", "/history/clear", "/health"],
    }


@app.get("/health")
async def health():
    vs, bm25, graph, hybrid, reranker, llm = get_components()
    return {
        "status": "healthy",
        "vector_store": vs.is_indexed(),
        "bm25": bm25.is_indexed(),
        "graph": graph.available if graph else False,
        "reranker": reranker.is_available(),
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "ingestion_status": _ingestion_status,
    }


@app.post("/ingest")
async def ingest_pdf(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest PDF and build all indexes."""
    global _ingestion_status

    def run_ingestion():
        global _ingestion_status
        try:
            _ingestion_status = {"status": "running", "message": "Parsing PDF...", "chunks": 0}
            vs, bm25, graph, hybrid, reranker, llm = get_components()

            # Parse PDF
            pipeline = PDFIngestionPipeline(request.pdf_path)
            chunks = pipeline.run(request.chunk_size, request.chunk_overlap)
            _ingestion_status["message"] = f"Parsed {len(chunks)} chunks. Building vector index..."

            # Index in vector store
            vs.index_chunks(chunks)
            _ingestion_status["message"] = "Vector index built. Building BM25 index..."

            # Build BM25
            bm25.build_index(chunks)
            _ingestion_status["message"] = "BM25 index built. Building knowledge graph..."

            # Build graph
            if request.rebuild_graph:
                graph.build_graph(chunks)

            _ingestion_status = {
                "status": "complete",
                "message": f"Ingestion complete. {len(chunks)} chunks indexed.",
                "chunks": len(chunks),
                "stats": pipeline.get_stats(),
            }
            logger.success(f"Ingestion complete: {len(chunks)} chunks")

        except Exception as e:
            _ingestion_status = {"status": "error", "message": str(e), "chunks": 0}
            logger.error(f"Ingestion failed: {e}")

    background_tasks.add_task(run_ingestion)
    return {"message": "Ingestion started in background", "status": "started"}


@app.get("/status")
async def ingestion_status():
    return _ingestion_status


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main RAG query endpoint."""
    global _conversation_history
    t_total = time.perf_counter()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    vs, bm25, graph, hybrid, reranker, llm = get_components()

    if not vs.is_indexed():
        raise HTTPException(
            status_code=503,
            detail="Document not yet ingested. Please POST /ingest first."
        )

    # ── Query Type Classification ───────────────────────────────────
    query_summary = get_query_summary(request.question)
    query_type = query_summary["query_type"]

    # ── Hybrid Retrieval ────────────────────────────────────────────
    t_retrieval = time.perf_counter()
    candidates, retrieval_meta = hybrid.retrieve(
        query=request.question,
        top_k=request.top_k,
        use_vector=request.use_vector,
        use_bm25=request.use_bm25,
        use_graph=request.use_graph,
    )
    retrieval_ms = round((time.perf_counter() - t_retrieval) * 1000, 1)

    if not candidates:
        return QueryResponse(
            answer="Information not found in the provided Physics document.",
            citations=[],
            chunks=[],
            query_type=query_type.value,
            retrieval_meta=retrieval_meta,
            latency={"total_ms": round((time.perf_counter() - t_total) * 1000, 1)},
            model=os.getenv("LLM_PROVIDER", "gemini"),
        )

    # ── Reranking ───────────────────────────────────────────────────
    t_rerank = time.perf_counter()
    if request.use_reranker:
        top_chunks = reranker.rerank(
            query=request.question,
            candidates=candidates,
            top_k=int(os.getenv("TOP_K_RERANK", "5")),
        )
    else:
        top_chunks = candidates[:5]
    rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 1)

    # ── Answer Generation ───────────────────────────────────────────
    t_gen = time.perf_counter()
    result = llm.generate(
        query=request.question,
        chunks=top_chunks,
        history=_conversation_history[-10:],
    )
    gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)

    total_ms = round((time.perf_counter() - t_total) * 1000, 1)

    # ── Update Conversation History ─────────────────────────────────
    _conversation_history.append({"role": "user", "content": request.question})
    _conversation_history.append({"role": "assistant", "content": result["answer"][:500]})
    if len(_conversation_history) > 20:
        _conversation_history = _conversation_history[-20:]

    # ── Observability Logging ───────────────────────────────────────
    observer.log_query(
        query=request.question,
        query_type=query_type,
        answer=result["answer"],
        citations=result["citations"],
        chunks=top_chunks,
        latency={
            "retrieval_ms": retrieval_ms,
            "reranking_ms": rerank_ms,
            "generation_ms": gen_ms,
            "total_ms": total_ms,
        },
        retrieval_meta=retrieval_meta,
        model=result["model"],
    )

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        chunks=top_chunks,
        query_type=query_type,
        retrieval_meta=retrieval_meta,
        latency={
            "retrieval_ms": retrieval_ms,
            "reranking_ms": rerank_ms,
            "generation_ms": gen_ms,
            "total_ms": total_ms,
        },
        model=result["model"],
    )


@app.post("/history/clear")
async def clear_history():
    global _conversation_history
    _conversation_history = []
    return {"message": "Conversation history cleared"}


@app.get("/history")
async def get_history():
    return {"history": _conversation_history, "turns": len(_conversation_history) // 2}


@app.get("/graph/chapter/{chapter_name}")
async def get_chapter_graph(chapter_name: str):
    """Get graph visualization data for a chapter."""
    _, _, graph, _, _, _ = get_components()
    if not graph or not graph.available:
        return {"nodes": [], "edges": [], "message": "Graph DB not available"}
    return graph.get_chapter_graph(chapter_name)


@app.get("/analytics")
async def get_analytics():
    """Session analytics: latency, grounding rate, query type distribution."""
    return observer.get_analytics()


@app.get("/analytics/failures")
async def get_failure_analysis():
    """Failure pattern analysis from session logs."""
    return observer.get_failure_analysis()


@app.post("/query/understand")
async def understand_query(request: QueryRequest):
    """Explain how a query would be classified and routed without executing it."""
    from app.retriever.query_understanding import get_query_summary
    return get_query_summary(request.question)

    vs, bm25, _, _, _, _ = get_components()
    return {
        "vector_store": vs.get_stats(),
        "bm25": {"indexed": bm25.is_indexed(), "chunks": len(bm25.chunks)},
        "conversation_turns": len(_conversation_history) // 2,
    }
