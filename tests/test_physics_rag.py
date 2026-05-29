"""
Test Suite for Physics Hybrid RAG
Tests: chunking, BM25, vector store, hybrid retriever, RRF, query classification, API

Run: pytest tests/ -v
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    """Minimal PhysicsChunk list for testing."""
    from app.ingestion.pdf_parser import PhysicsChunk
    return [
        PhysicsChunk(
            chunk_id="ch01_p001_c0000",
            page=1,
            chapter="Electric Charges and Fields",
            chapter_num=1,
            heading="Electric Charges",
            subheading="1.1 Introduction",
            content="Electric charge is a fundamental property of matter. "
                    "Coulomb's law states that F = kq1q2/r². "
                    "Like charges repel and unlike charges attract.",
            formulas=["F = kq1q2/r²", "E = F/q"],
            tables=[],
            chunk_type="text",
        ),
        PhysicsChunk(
            chunk_id="ch01_p005_c0001",
            page=5,
            chapter="Electric Charges and Fields",
            chapter_num=1,
            heading="Electric Field",
            subheading="1.4 Electric Field",
            content="The electric field is defined as the force per unit positive charge. "
                    "E = F/q. The SI unit of electric field is N/C or V/m. "
                    "Electric field lines originate from positive charges.",
            formulas=["E = F/q", "E = kq/r²"],
            tables=[],
            chunk_type="formula",
        ),
        PhysicsChunk(
            chunk_id="ch02_p045_c0010",
            page=45,
            chapter="Electrostatic Potential and Capacitance",
            chapter_num=2,
            heading="Electric Potential",
            subheading="2.1 Introduction",
            content="Electric potential is the work done per unit charge in bringing "
                    "a test charge from infinity to a point. V = kq/r. "
                    "Potential difference drives current flow.",
            formulas=["V = kq/r", "V = W/q"],
            tables=[],
            chunk_type="text",
        ),
        PhysicsChunk(
            chunk_id="ch03_p091_c0020",
            page=91,
            chapter="Current Electricity",
            chapter_num=3,
            heading="Ohm's Law",
            subheading="3.3 Ohm's Law",
            content="Ohm's law states that V = IR where V is voltage, I is current, "
                    "and R is resistance. The SI unit of resistance is ohm (Ω). "
                    "Resistivity is an intrinsic property of the material.",
            formulas=["V = IR", "R = ρL/A"],
            tables=[],
            chunk_type="text",
        ),
    ]


# ── Ingestion Tests ───────────────────────────────────────────────────────────

class TestPDFParser:
    def test_formula_extraction(self):
        from app.ingestion.pdf_parser import PDFIngestionPipeline
        pipeline = PDFIngestionPipeline("dummy.pdf")
        text = "The force is F = kq1q2/r². Also E = F/q and V = IR."
        formulas = pipeline.extract_formulas(text)
        assert len(formulas) > 0

    def test_chapter_inference(self):
        from app.ingestion.pdf_parser import PDFIngestionPipeline
        pipeline = PDFIngestionPipeline("dummy.pdf")
        num, name = pipeline.infer_chapter(10, "This chapter covers electrostatics")
        assert isinstance(num, int)
        assert isinstance(name, str)

    def test_heading_extraction(self):
        from app.ingestion.pdf_parser import PDFIngestionPipeline
        pipeline = PDFIngestionPipeline("dummy.pdf")
        text = "ELECTRIC FIELD\n2.3 Gauss Law\nSome content here."
        heading, subheading = pipeline.extract_headings(text)
        assert isinstance(heading, str)
        assert isinstance(subheading, str)

    def test_chunk_structure(self, sample_chunks):
        assert len(sample_chunks) == 4
        for chunk in sample_chunks:
            assert chunk.chunk_id
            assert chunk.page > 0
            assert chunk.chapter
            assert len(chunk.content) > 10


# ── BM25 Tests ────────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_build_and_search(self, sample_chunks):
        from app.retriever.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(index_path="/tmp/test_bm25.pkl")
        bm25.build_index(sample_chunks)
        assert bm25.is_indexed()

        results = bm25.search("Coulomb's law electric force", top_k=3)
        assert len(results) > 0
        assert "chunk_id" in results[0]
        assert "bm25_score" in results[0]
        assert results[0]["bm25_score"] >= 0

    def test_search_returns_formula_chunk(self, sample_chunks):
        from app.retriever.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(index_path="/tmp/test_bm25_2.pkl")
        bm25.build_index(sample_chunks)

        results = bm25.search("Ohm's law V=IR resistance", top_k=5)
        chunk_ids = [r["chunk_id"] for r in results]
        # Ohm's law chunk should be in results
        assert any("ch03" in cid for cid in chunk_ids)

    def test_empty_query(self, sample_chunks):
        from app.retriever.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(index_path="/tmp/test_bm25_3.pkl")
        bm25.build_index(sample_chunks)
        results = bm25.search("", top_k=3)
        assert results == []

    def test_tokenization(self):
        from app.retriever.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever()
        tokens = bm25._tokenize("E=mc² where m is mass and c=3×10⁸")
        assert len(tokens) > 0
        assert "e" in tokens or "mc" in tokens


# ── Query Understanding Tests ─────────────────────────────────────────────────

class TestQueryUnderstanding:
    def test_definition_classification(self):
        from app.retriever.query_understanding import classify_query, QueryType
        qtype, profile = classify_query("What is electric charge?")
        assert qtype == QueryType.DEFINITION

    def test_formula_classification(self):
        from app.retriever.query_understanding import classify_query, QueryType
        qtype, profile = classify_query("State the formula for Coulomb's law")
        assert qtype == QueryType.FORMULA

    def test_numerical_classification(self):
        from app.retriever.query_understanding import classify_query, QueryType
        qtype, profile = classify_query("Calculate the force between two charges")
        assert qtype == QueryType.NUMERICAL

    def test_comparison_classification(self):
        from app.retriever.query_understanding import classify_query, QueryType
        qtype, profile = classify_query("What is the difference between electric potential and potential energy?")
        assert qtype == QueryType.COMPARISON

    def test_multi_hop_classification(self):
        from app.retriever.query_understanding import classify_query, QueryType
        qtype, profile = classify_query("What is the relationship between electric field and potential?")
        assert qtype == QueryType.MULTI_HOP

    def test_profile_has_required_keys(self):
        from app.retriever.query_understanding import classify_query
        _, profile = classify_query("Explain Faraday's law")
        assert "vector" in profile
        assert "bm25" in profile
        assert "graph" in profile
        assert "rationale" in profile

    def test_query_expansion(self):
        from app.retriever.query_understanding import expand_query, QueryType
        expanded = expand_query("Explain Coulomb's law", QueryType.FORMULA)
        assert len(expanded) > len("Explain Coulomb's law")

    def test_get_query_summary(self):
        from app.retriever.query_understanding import get_query_summary
        summary = get_query_summary("What is Ohm's law?")
        assert "query_type" in summary
        assert "retrieval_profile" in summary
        assert "expanded_query" in summary


# ── RRF Fusion Tests ──────────────────────────────────────────────────────────

class TestRRFFusion:
    def _rrf(self, ranked_lists, k=60):
        """Inline RRF for testing without chromadb dependency."""
        from collections import defaultdict
        rrf_scores = defaultdict(float)
        chunk_data = {}
        for ranked_list in ranked_lists:
            for rank, hit in enumerate(ranked_list, start=1):
                cid = hit["chunk_id"]
                rrf_scores[cid] += 1.0 / (k + rank)
                if cid not in chunk_data:
                    chunk_data[cid] = hit
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for cid, score in sorted_chunks:
            h = chunk_data[cid].copy()
            h["rrf_score"] = round(score, 6)
            result.append(h)
        return result

    def test_rrf_scores_are_positive(self):
        list1 = [{"chunk_id": "a", "content": "x", "metadata": {}, "vector_score": 0.9}]
        list2 = [{"chunk_id": "a", "content": "x", "metadata": {}, "bm25_score": 0.8},
                 {"chunk_id": "b", "content": "y", "metadata": {}, "bm25_score": 0.5}]

        fused = self._rrf([list1, list2])
        assert len(fused) == 2
        assert fused[0]["rrf_score"] > 0
        assert fused[0]["chunk_id"] == "a"

    def test_rrf_deduplicates(self):
        hit = {"chunk_id": "dup", "content": "x", "metadata": {}}
        fused = self._rrf([[hit], [hit]])
        chunk_ids = [f["chunk_id"] for f in fused]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_rrf_k60_formula(self):
        hits = [
            {"chunk_id": "c1", "content": "", "metadata": {}},
            {"chunk_id": "c2", "content": "", "metadata": {}},
        ]
        fused = self._rrf([hits])
        assert abs(fused[0]["rrf_score"] - 1/61) < 1e-5
        assert abs(fused[1]["rrf_score"] - 1/62) < 1e-5


# ── LLM Client Tests ──────────────────────────────────────────────────────────

class TestLLMClient:
    def test_empty_chunks_returns_not_found(self):
        from app.api.llm_client import LLMClient
        with patch.object(LLMClient, '_setup'):
            client = LLMClient.__new__(LLMClient)
            client.provider = "gemini"
            result = client.generate("What is X?", chunks=[])
        assert "not found" in result["answer"].lower()

    def test_format_context(self):
        from app.api.llm_client import format_context
        chunks = [{
            "content": "Test content",
            "metadata": {"page": 5, "chapter": "Ch1", "subheading": "Sec 1.1", "formulas": "F=ma"},
            "chunk_id": "test_001",
        }]
        ctx = format_context(chunks)
        assert "Page 5" in ctx
        assert "Ch1" in ctx
        assert "Test content" in ctx

    def test_format_history(self):
        from app.api.llm_client import format_history
        history = [
            {"role": "user", "content": "What is charge?"},
            {"role": "assistant", "content": "Charge is a property of matter."},
        ]
        result = format_history(history)
        assert "user" in result.lower() or "User" in result
        assert "charge" in result.lower()

    def test_extract_citations(self):
        from app.api.llm_client import LLMClient
        with patch.object(LLMClient, '_setup'):
            client = LLMClient.__new__(LLMClient)
        chunks = [
            {"chunk_id": "a", "metadata": {"page": 10, "chapter": "Ch1", "subheading": ""}},
            {"chunk_id": "b", "metadata": {"page": 10, "chapter": "Ch1", "subheading": ""}},
            {"chunk_id": "c", "metadata": {"page": 15, "chapter": "Ch2", "subheading": ""}},
        ]
        citations = client._extract_citations(chunks)
        # page 10+Ch1 and page 15+Ch2 → 2 unique citations
        assert len(citations) == 2


# ── Observability Tests ───────────────────────────────────────────────────────

class TestObservability:
    def test_log_and_analytics(self):
        from app.api.observability import RAGObserver
        obs = RAGObserver()

        obs.log_query(
            query="What is Ohm's law?",
            query_type="definition",
            answer="Ohm's law states V=IR [Page 91, Chapter: Current Electricity]",
            citations=[{"page": 91, "chapter": "Current Electricity"}],
            chunks=[{"chunk_id": "ch03_p091_c0020", "metadata": {"page": 91},
                     "vector_score": 0.8, "bm25_score": 0.6,
                     "rrf_score": 0.016, "reranker_score": 0.95}],
            latency={"total_ms": 1250, "retrieval_ms": 320,
                     "reranking_ms": 180, "generation_ms": 750},
            retrieval_meta={"timing": {"vector_ms": 120, "bm25_ms": 80},
                            "vector_results": [{}], "bm25_results": [{}], "graph_results": []},
            model="gemini",
        )

        analytics = obs.get_analytics()
        assert analytics["total_queries"] >= 1
        assert "grounding_rate_pct" in analytics
        assert "latency" in analytics

    def test_failure_analysis_structure(self):
        from app.api.observability import RAGObserver
        obs = RAGObserver()
        fa = obs.get_failure_analysis()
        assert "message" in fa or "not_found_rate_pct" in fa
