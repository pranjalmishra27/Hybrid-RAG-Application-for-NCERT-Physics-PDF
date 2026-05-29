"""
Vector Database Module
Uses ChromaDB with BAAI/bge-small-en-v1.5 embeddings for semantic retrieval.
"""

import os
import pickle
from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np

from app.ingestion.pdf_parser import PhysicsChunk


class VectorStore:
    """
    ChromaDB-backed vector store with BGE embeddings.
    
    BGE-small-en-v1.5 chosen for:
    - Strong performance on retrieval benchmarks (MTEB)
    - Small footprint (133MB) vs larger alternatives
    - Optimized for asymmetric retrieval (query vs passage)
    - Free / local inference, no API costs
    """

    COLLECTION_NAME = "physics_chunks"

    def __init__(self,
                 persist_dir: str = "./data/chroma_db",
                 model_name: str = "BAAI/bge-small-en-v1.5"):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self._embedding_model: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _encode(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        BGE requires instruction prefix for queries.
        For passages: no prefix needed.
        For queries: prepend 'Represent this sentence for searching relevant passages:'
        """
        if is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def index_chunks(self, chunks: List[PhysicsChunk], batch_size: int = 100):
        """Index all chunks into ChromaDB."""
        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB...")

        # Clear existing
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
        except Exception:
            pass

        all_ids = []
        all_texts = []
        all_metadatas = []

        for chunk in chunks:
            all_ids.append(chunk.chunk_id)
            all_texts.append(chunk.content)
            all_metadatas.append({
                "page": chunk.page,
                "chapter": chunk.chapter,
                "chapter_num": chunk.chapter_num,
                "heading": chunk.heading,
                "subheading": chunk.subheading,
                "chunk_type": chunk.chunk_type,
                "formulas": " | ".join(chunk.formulas) if chunk.formulas else "",
                "source": chunk.source,
            })

        # Batch encode and upsert
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            batch_meta = all_metadatas[i:i + batch_size]

            embeddings = self._encode(batch_texts, is_query=False)

            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            logger.debug(f"Indexed batch {i // batch_size + 1}/{(len(all_texts) + batch_size - 1) // batch_size}")

        logger.success(f"ChromaDB indexing complete. Collection size: {self.collection.count()}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic similarity search."""
        query_embedding = self._encode([query], is_query=True)[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        if not results["ids"][0]:
            return hits

        for i, chunk_id in enumerate(results["ids"][0]):
            similarity = 1 - results["distances"][0][i]  # cosine: distance -> similarity
            hits.append({
                "chunk_id": chunk_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "vector_score": round(float(similarity), 4),
                "retrieval_source": "vector",
            })

        return hits

    def is_indexed(self) -> bool:
        """Check if index already exists."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def get_stats(self) -> Dict:
        return {
            "collection": self.COLLECTION_NAME,
            "total_documents": self.collection.count(),
            "persist_dir": self.persist_dir,
            "embedding_model": self.model_name,
        }
