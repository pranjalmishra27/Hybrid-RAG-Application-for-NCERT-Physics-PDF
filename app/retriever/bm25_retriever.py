"""
BM25 Keyword Search Module
Supports exact keyword matching, formula lookup, definitions, numerical concepts.

BM25 (Best Match 25) chosen over TF-IDF because:
- Penalizes document length (prevents long docs from dominating)
- Saturates term frequency (diminishing returns for repeated terms)
- Standard baseline for keyword retrieval (used in Elasticsearch internally)
- Excellent for physics: formula names, constants, exact terms
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from loguru import logger

from app.ingestion.pdf_parser import PhysicsChunk


class BM25Retriever:
    def __init__(self, index_path: str = "./data/bm25_index.pkl"):
        self.index_path = Path(index_path)
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[PhysicsChunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Physics-aware tokenization.
        Preserves: formula notation, numbers, units, Greek letters transliterated.
        """
        # Lowercase
        text = text.lower()

        # Preserve physics notation: E=mc^2 -> keep as tokens
        text = re.sub(r'([a-z])=', r'\1 = ', text)

        # Split on whitespace and punctuation, keep alphanumeric + physics chars
        tokens = re.findall(r'[a-z0-9αβγδεζηθικλμνξοπρστυφχψω\.\-\+\^\/\*²³]+', text)

        # Remove very short tokens (noise) but keep single-letter physics vars
        tokens = [t for t in tokens if len(t) >= 1]

        return tokens

    def build_index(self, chunks: List[PhysicsChunk]):
        """Build BM25 index from chunks."""
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        self.chunks = chunks

        # Tokenize: combine content + formulas for richer matching
        self.tokenized_corpus = []
        for chunk in chunks:
            text = chunk.content
            if chunk.formulas:
                text += " " + " ".join(chunk.formulas)
            if chunk.subheading:
                text += " " + chunk.subheading  # boost heading terms
                text += " " + chunk.subheading  # double weight
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.success(f"BM25 index built. Vocabulary size: {len(self.bm25.idf)}")

        # Persist
        self._save()

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunks": self.chunks,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)
        logger.info(f"BM25 index saved to {self.index_path}")

    def load(self) -> bool:
        """Load persisted BM25 index."""
        if not self.index_path.exists():
            return False
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]
            self.tokenized_corpus = data["tokenized_corpus"]
            logger.info(f"BM25 index loaded: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 keyword search."""
        if self.bm25 is None:
            logger.warning("BM25 index not built. Returning empty results.")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0

        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue

            chunk = self.chunks[idx]
            hits.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": {
                    "page": chunk.page,
                    "chapter": chunk.chapter,
                    "chapter_num": chunk.chapter_num,
                    "heading": chunk.heading,
                    "subheading": chunk.subheading,
                    "chunk_type": chunk.chunk_type,
                    "formulas": " | ".join(chunk.formulas) if chunk.formulas else "",
                    "source": chunk.source,
                },
                "bm25_score": round(score / max(max_score, 1e-9), 4),  # normalize 0-1
                "bm25_raw_score": round(score, 4),
                "retrieval_source": "bm25",
            })

        return hits

    def is_indexed(self) -> bool:
        return self.bm25 is not None and len(self.chunks) > 0
