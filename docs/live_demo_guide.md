# Live Demo Guide — Physics Hybrid RAG

## Pre-Demo Checklist (5 minutes before)

```bash
# 1. Verify all services running
docker-compose ps

# 2. Check API health
curl http://localhost:8000/health | python -m json.tool

# 3. Check ingestion complete
curl http://localhost:8000/status

# 4. Open browser tabs
#    - http://localhost:8501  (Streamlit frontend)
#    - http://localhost:8000/docs  (FastAPI Swagger)
#    - http://localhost:7474  (Neo4j Browser) [optional]
```

---

## Demo Script (15 minutes)

### Scene 1: Architecture Overview (2 min)

Open `docs/architecture_diagram.png` (or show the interactive diagram).

**Say**: "The system uses three complementary retrieval strategies — dense vector
search for semantic similarity, BM25 for exact keyword matching, and a Neo4j
knowledge graph for concept relationships. These are fused using Reciprocal Rank
Fusion before a cross-encoder reranker selects the top 5 chunks for the LLM."

---

### Scene 2: Simple Definition Query (2 min)

Type: **"What is electric charge?"**

**Point to**:
- Answer panel: grounded response citing page and chapter
- Citations panel: "Page 1, Chapter: Electric Charges and Fields"
- Latency panel: show retrieval breakdown (vector vs BM25 vs graph)
- Query Type badge: "DEFINITION"

**Explain**: "For definition queries, BM25 gets extra weight because exact term
matching is more reliable than semantic similarity for looking up definitions."

---

### Scene 3: Formula Query (3 min)

Type: **"State Coulomb's Law with its mathematical formula."**

**Point to**:
- Answer: should include F = kq₁q₂/r² with citation
- Retrieved Chunks tab: show the chunk containing the formula
- Explainability scores: note that BM25 score is high (exact keyword hit)
- Graph panel: show the Coulomb → Law → Formula path

**Explain**: "Notice how BM25 retrieves the exact formula text while the graph
retrieval finds it via the knowledge path: Coulomb → Law → Formula → Electric
Charges chapter."

---

### Scene 4: Toggle Retrieval Sources (2 min)

1. In sidebar, **disable BM25**, re-run Coulomb query
2. Show that answer quality may decrease slightly (formula may be less prominent)
3. **Re-enable BM25**, disable Graph, run: **"How does Coulomb's Law relate to Gauss's Law?"**
4. Re-enable all, run same query — show graph adds relationship context

**Explain**: "This demonstrates why hybrid retrieval beats any single method.
BM25 wins for exact terms; vector wins for semantic similarity; graph wins for
concept relationships."

---

### Scene 5: Follow-up / Multi-turn (2 min)

First query: **"Explain Faraday's Law of electromagnetic induction."**

Then immediately: **"What are its limitations?"**

**Point to**: The system understands "its" refers to Faraday's Law from context.

**Explain**: "The system maintains 5 turns of conversation history. The second
query is resolved using the prior context — 'its limitations' is understood as
'Faraday's Law's limitations' without needing to repeat the subject."

---

### Scene 6: Out-of-Scope Query (1 min)

Type: **"What is the speed of sound in water?"**

**Expected response**: "Information not found in the provided Physics document."

**Explain**: "The system is strictly grounded. It does not hallucinate or use
general physics knowledge outside the PDF. This is the anti-hallucination
mechanism — strict system prompt + context-only generation."

---

### Scene 7: Latency & Observability (2 min)

- Switch to Latency tab, show the timing breakdown
- Navigate to `http://localhost:8000/docs`, show `/stats` and `/history` endpoints
- Run: `GET /stats` — show ChromaDB and BM25 index stats

**Explain**: "Every query is logged to a JSONL file. LangSmith traces are available
when the API key is configured. The analytics endpoint shows p50/p95 latency,
grounding rate, and retrieval hit rates across the session."

---

### Scene 8: Reviewer Questions (handle live)

Be ready for:

| Question | Key talking points |
|----------|--------------------|
| "Why ChromaDB over Pinecone?" | Local dev, no API costs, persistent; Pinecone for production scale |
| "Why BGE over OpenAI embeddings?" | Free, local, no API dependency, comparable quality for English physics |
| "Why k=60 in RRF?" | Empirically established in RRF paper (Cormack et al., 2009); reduces outlier rank impact |
| "What if Neo4j is down?" | System degrades gracefully — vector + BM25 still work, graph results return empty |
| "How do you prevent hallucination?" | System prompt with explicit rules + "not found" fallback + low temperature |
| "What's your chunking strategy?" | Section-aware (split on section headings) + sliding window (900t, 175 overlap). Tradeoff: preserves section context vs may split long derivations |
| "How would you scale this?" | Qdrant cluster for vector, Elasticsearch for BM25, Neo4j Aura, Redis cache, async parallel retrieval |

---

## Troubleshooting During Demo

| Issue | Fix |
|-------|-----|
| "API not reachable" | `docker-compose restart backend` |
| "Document not yet ingested" | Click "🚀 Ingest PDF" in sidebar |
| Slow response (>10s) | Reranker cold-starting — subsequent queries will be faster |
| Graph shows 0 results | Neo4j container not healthy — queries still work via vector+BM25 |
| Answer is "not found" | PDF may not cover that topic, or chunking missed it |

---

## Backup: API Demo (if Streamlit unavailable)

```bash
# Quick definition query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Coulomb'\''s Law?", "top_k": 10}' | python -m json.tool

# Formula query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "State the formula for electric field", "use_graph": true}' | python -m json.tool

# Check history
curl http://localhost:8000/history | python -m json.tool

# Stats
curl http://localhost:8000/stats | python -m json.tool
```
