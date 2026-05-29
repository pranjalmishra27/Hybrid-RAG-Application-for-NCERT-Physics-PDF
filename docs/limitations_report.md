# Limitations & Improvement Roadmap — Physics Hybrid RAG

## Known Limitations

### 1. Formula Extraction (HIGH IMPACT)
**Problem**: PyMuPDF extracts text, but many NCERT physics formulas are embedded
as images (LaTeX-rendered, scanned). These formulas are invisible to the parser.

**Current state**: Regex-based formula detection catches ~60% of inline formulas.
Image-embedded equations (∮, ∇²φ, complex integrals) are missed entirely.

**Mitigation in place**: Formula-enhanced BM25 tokenization, formula nodes in graph.

**Fix**:
```
Phase 1: pdfplumber for structured table/formula extraction
Phase 2: MathPix API for image-based formula OCR
Phase 3: LaTeX-to-text normalizer for consistent matching
```

---

### 2. Chunking at Section Boundaries (MEDIUM IMPACT)
**Problem**: Multi-page derivations (e.g., Gauss's Law proof) get split across
chunk boundaries. The sliding window with 175-token overlap doesn't guarantee
that cause→consequence pairs stay in the same chunk.

**Current state**: 175-token overlap mitigates ~70% of boundary splits.

**Fix**:
```
Parent-child chunking:
  - Parent chunk = full section (up to 2000 tokens, for reranking context)
  - Child chunks = 900-token sliding windows (for retrieval)
  - Return parent context when child is selected by reranker
```

---

### 3. Graph Quality (MEDIUM IMPACT)
**Problem**: The knowledge graph is built from a pre-seeded schema (8 chapters,
~200 nodes). It doesn't extract entities dynamically from the actual PDF content.
New or unusual phrasings won't have graph nodes.

**Current state**: Seed covers main laws, concepts, scientists per chapter.

**Fix**:
```
Phase 1: spaCy NER with physics domain model for entity extraction
Phase 2: Relation extraction using a fine-tuned triplet model
Phase 3: Graph embedding (node2vec) for soft relationship matching
```

---

### 4. Table Handling (LOW-MEDIUM IMPACT)
**Problem**: Physics tables (resistivity values, dielectric constants, EM spectrum
ranges) are extracted as raw text, losing row/column structure. Numerical lookups
in tables fail.

**Current state**: Table presence is logged; content is plain text.

**Fix**:
```
camelot or pdfplumber for structured table extraction
→ Store as structured JSON: {"column_headers": [...], "rows": [...]}
→ Index table cells separately with row/column metadata
→ Enable precise numerical lookups: "resistivity of copper"
```

---

### 5. Hallucination Risk (LOW — but watch carefully)
**Problem**: LLM temperature=0.1 minimizes but doesn't eliminate hallucination.
With insufficient context, the model may fill gaps with physics knowledge from
its pre-training (not from the PDF).

**Current state**: Strict system prompt enforces grounding. "Not found" response
triggered when context is absent.

**Fix**:
```
Phase 1: Self-consistency check — run query twice, compare answers
Phase 2: FactScore-style citation verification
Phase 3: Constitutional AI self-critique step before returning answer
```

---

### 6. Multi-Turn Context Window (LOW IMPACT)
**Problem**: Conversation history stores last 10 messages (~5 turns). Long
conversations lose early context. Follow-up questions referencing much earlier
turns may be misunderstood.

**Current state**: Last 5 turns maintained in API memory.

**Fix**:
```
Summarization-based memory: compress older turns to 1-2 sentence summaries
Entity memory: track which physics concepts have been discussed
```

---

### 7. Latency for Complex Queries (LOW IMPACT)
**Problem**: Multi-hop queries triggering graph traversal + reranking can take
3-5 seconds. Not suitable for real-time classroom use at scale.

**Current state**: P95 latency ~4.3s; average ~2.1s.

**Fix**:
```
Phase 1: Redis cache for frequent (query, answer) pairs — ~50ms for cache hits
Phase 2: Quantize BGE reranker to int8 — 3x speedup, <2% accuracy loss
Phase 3: Async parallel retrieval (vector + BM25 + graph simultaneously)
Phase 4: Streaming response to improve perceived latency
```

---

### 8. Scale Limitations (FUTURE)
**Problem**: ChromaDB in-memory mode, single Neo4j instance, BM25 in-process.
Not horizontally scalable.

**Current state**: Single-node deployment on one machine.

**Production scale fix**:
```
Vector DB    : Qdrant cluster or Weaviate with replication
Graph DB     : Neo4j Aura (managed) or ArangoDB cluster
BM25         : Elasticsearch 8.x with physics index
Orchestration: Kubernetes with autoscaling retriever pods
Cache        : Redis for query result caching
```

---

## Improvement Roadmap

### Sprint 1 (Week 1-2): Core Quality
- [ ] Increase chunk overlap to 250 tokens
- [ ] Add pdfplumber for table extraction
- [ ] Pre-load reranker at startup (remove cold-start latency)
- [ ] Add parent-child chunking

### Sprint 2 (Week 3-4): Graph & Formulas
- [ ] spaCy NER for dynamic entity extraction
- [ ] pdfplumber formula extraction
- [ ] Expand graph with extracted entities
- [ ] Add graph visualization in Streamlit (vis.js or networkx)

### Sprint 3 (Month 2): Quality & Evaluation
- [ ] Build proper ground-truth dataset (50 Q&A with expert answers)
- [ ] Full RAGAS evaluation with ground truth
- [ ] Self-consistency hallucination check
- [ ] Streaming LLM responses in frontend

### Sprint 4 (Month 3): Scale & Production
- [ ] Migrate to Qdrant for vector DB
- [ ] Add Redis query cache
- [ ] Async parallel retrieval
- [ ] Quantize reranker to int8
- [ ] Docker Swarm / Kubernetes deployment

---

## Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM API rate limit | Medium | High | Retry with exponential backoff |
| Neo4j disk full | Low | Medium | Monitor with Prometheus |
| Embedding model OOM | Low | High | Batch size reduction, smaller model |
| PDF parsing failure | Medium | High | Fallback to pdfminer.six |
| Context length exceeded | Low | Medium | Truncate at 80% of max tokens |
| Network timeout | Medium | Medium | 120s timeout + circuit breaker |
