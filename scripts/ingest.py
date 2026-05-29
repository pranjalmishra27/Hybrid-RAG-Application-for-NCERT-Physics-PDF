#!/usr/bin/env python3
"""
Ingestion CLI Script
One-command setup: parses PDF, builds vector index, BM25, and knowledge graph.

Usage:
    python scripts/ingest.py --pdf ./data/ncert_physics_part1.pdf
    python scripts/ingest.py --pdf ./data/ncert_physics_part1.pdf --skip-graph
    python scripts/ingest.py --pdf ./data/ncert_physics_part1.pdf --chunk-size 900 --overlap 175
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from app.ingestion.pdf_parser import PDFIngestionPipeline
from app.vector_db.chroma_store import VectorStore
from app.retriever.bm25_retriever import BM25Retriever
from app.graph.neo4j_graph import GraphRetriever


def run_ingestion(
    pdf_path: str,
    chunk_size: int = 900,
    chunk_overlap: int = 175,
    skip_graph: bool = False,
    force_rebuild: bool = False,
):
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("PHYSICS RAG — INGESTION PIPELINE")
    logger.info("=" * 60)

    # ── 1. Parse PDF ──────────────────────────────────────────────
    logger.info(f"\n[1/4] Parsing PDF: {pdf_path}")
    pipeline = PDFIngestionPipeline(pdf_path)
    chunks = pipeline.run(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats = pipeline.get_stats()

    logger.info(f"  Pages parsed   : {stats['total_pages']}")
    logger.info(f"  Chunks created : {stats['total_chunks']}")
    logger.info(f"  Chapters found : {stats['chapters']}")
    logger.info(f"  Avg chunk words: {stats['avg_chunk_words']}")
    logger.info(f"  Formula chunks : {stats['formula_chunks']}")

    # Save chunk dump for inspection
    os.makedirs("./data", exist_ok=True)
    pipeline.save_chunks("./data/chunks_debug.json")
    logger.info("  Chunk dump saved: ./data/chunks_debug.json")

    # ── 2. Vector Store ───────────────────────────────────────────
    logger.info(f"\n[2/4] Building ChromaDB vector index...")
    vector_store = VectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )

    if not force_rebuild and vector_store.is_indexed():
        logger.info("  Vector index already exists. Use --force to rebuild.")
    else:
        t0 = time.time()
        vector_store.index_chunks(chunks)
        logger.info(f"  Done in {time.time()-t0:.1f}s — {vector_store.get_stats()['total_documents']} vectors")

    # ── 3. BM25 Index ─────────────────────────────────────────────
    logger.info(f"\n[3/4] Building BM25 keyword index...")
    bm25 = BM25Retriever(
        index_path=os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
    )

    if not force_rebuild and bm25.load():
        logger.info(f"  BM25 index loaded: {len(bm25.chunks)} chunks")
    else:
        t0 = time.time()
        bm25.build_index(chunks)
        logger.info(f"  Done in {time.time()-t0:.1f}s")

    # ── 4. Knowledge Graph ────────────────────────────────────────
    if skip_graph:
        logger.info(f"\n[4/4] Skipping graph build (--skip-graph)")
    else:
        logger.info(f"\n[4/4] Building Neo4j knowledge graph...")
        graph = GraphRetriever(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "physics_rag_2024"),
        )
        if graph.available:
            t0 = time.time()
            graph.build_graph(chunks)
            graph.close()
            logger.info(f"  Graph built in {time.time()-t0:.1f}s")
        else:
            logger.warning("  Neo4j not available — start with: docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/physics_rag_2024 neo4j:5.18-community")

    # ── Summary ───────────────────────────────────────────────────
    total_time = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time    : {total_time:.1f}s")
    logger.info(f"Chunks indexed: {len(chunks)}")
    logger.info(f"Vector DB     : {os.getenv('CHROMA_PERSIST_DIR', './data/chroma_db')}")
    logger.info(f"BM25 index    : {os.getenv('BM25_INDEX_PATH', './data/bm25_index.pkl')}")
    logger.info("\nNext steps:")
    logger.info("  Start backend : uvicorn app.api.main:app --port 8000")
    logger.info("  Start frontend: streamlit run app/frontend/streamlit_app.py")
    logger.info("  Or both via  : docker-compose up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics RAG Ingestion Pipeline")
    parser.add_argument("--pdf", default="./data/ncert_physics_part1.pdf", help="Path to NCERT Physics PDF")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in tokens (default: 900)")
    parser.add_argument("--overlap", type=int, default=175, help="Chunk overlap in tokens (default: 175)")
    parser.add_argument("--skip-graph", action="store_true", help="Skip Neo4j graph build")
    parser.add_argument("--force", action="store_true", help="Force rebuild all indexes")
    args = parser.parse_args()

    run_ingestion(
        pdf_path=args.pdf,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        skip_graph=args.skip_graph,
        force_rebuild=args.force,
    )
