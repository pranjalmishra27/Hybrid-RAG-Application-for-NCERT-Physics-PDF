# ⚛️ Physics Hybrid RAG — NCERT Class 12 Physics

> **Production-grade Hybrid Retrieval-Augmented Generation system for NCERT Class 12 Physics Part 1.**
> Built for the VANCO AI Solution Architect Assessment.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange?style=flat-square)](https://www.trychroma.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.18-008CC1?style=flat-square&logo=neo4j)](https://neo4j.com)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Query Preprocessing   │  ← Query type classification
│   (Query Understanding) │    (definition/formula/numerical/
└──────────┬──────────────┘     comparison/conceptual/multi-hop)
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              HYBRID RETRIEVER                        │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ Vector      │ │ BM25 Keyword │ │ Graph        │  │
│  │ (ChromaDB)  │ │ (rank-bm25)  │ │ (Neo4j)      │  │
│  │ BGE-small   │ │ Exact match  │ │ Concept graph│  │
│  └──────┬──────┘ └──────┬───────┘ └──────┬───────┘  │
│         └───────────────┴────────────────┘          │
│                         │                            │
│              ┌──────────▼──────────┐                 │
│              │  RRF Fusion         │                 │
│              │  score = Σ 1/(60+r) │                 │
│              └──────────┬──────────┘                 │
└─────────────────────────┼────────────────────────────┘
                          │
                          ▼ Top 20 candidates
              ┌───────────────────────┐
              │  BGE Reranker         │
              │  (cross-encoder)      │
              └───────────┬───────────┘
                          │ Top 5 chunks
                          ▼
              ┌───────────────────────┐
              │  Prompt Construction  │
              │  + History (last 5)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  LLM (Gemini 2.5 Flash│
              │   or OpenAI GPT-4o)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Grounded Answer      │
              │  + Citation Generator │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Streamlit Frontend   │
              │  Explainability Panel │
              └───────────────────────┘
```

---

## 🚀 Quick Start (Docker — Recommended)

### Prerequisites
- Docker + Docker Compose
- GEMINI_API_KEY or OPENAI_API_KEY
- NCERT Physics Class 12 Part 1 PDF (place in `./data/`)

### 1. Clone and Configure

```bash
git clone https://github.com/your-username/physics-rag.git
cd physics-rag

# Copy and fill in your API keys
cp .env.example .env
nano .env   # Add GEMINI_API_KEY or OPENAI_API_KEY
```

### 2. Add PDF

```bash
# Place the NCERT PDF in the data directory
cp ~/Downloads/ncert_physics_12_part1.pdf ./data/ncert_physics_part1.pdf
```

### 3. Start All Services

```bash
docker-compose up --build
```

Services started:
| Service | URL |
|---------|-----|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| Neo4j Browser | http://localhost:7474 |
| API Docs | http://localhost:8000/docs |

### 4. Ingest the PDF

Via Streamlit sidebar → **"🚀 Ingest PDF"** button,
or via API:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "./data/ncert_physics_part1.pdf"}'
```

### 5. Ask Questions

Open http://localhost:8501 and start asking!

---

## 🏃 Local Development (Without Docker)

### Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
```

### Start Neo4j (optional, for graph retrieval)

```bash
# Using Docker for just Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/physics_rag_2024 \
  neo4j:5.18-community
```

### Start Backend

```bash
uvicorn app.api.main:app --reload --port 8000
```

### Start Frontend

```bash
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

---

## 📂 Project Structure

```
physics-rag/
│
├── app/
│   ├── api/
│   │   ├── main.py              # FastAPI endpoints
│   │   └── llm_client.py        # Gemini / OpenAI LLM client
│   │
│   ├── ingestion/
│   │   └── pdf_parser.py        # PDF parsing + section-aware chunking
│   │
│   ├── vector_db/
│   │   └── chroma_store.py      # ChromaDB + BGE embeddings
│   │
│   ├── retriever/
│   │   ├── bm25_retriever.py    # BM25 keyword search
│   │   └── hybrid_retriever.py  # RRF fusion + query classification
│   │
│   ├── graph/
│   │   └── neo4j_graph.py       # Knowledge graph build + retrieval
│   │
│   ├── reranker/
│   │   └── reranker.py          # BGE cross-encoder reranker
│   │
│   ├── evaluation/
│   │   └── ragas_eval.py        # 50 benchmark questions + RAGAS metrics
│   │
│   └── frontend/
│       └── streamlit_app.py     # Full Streamlit UI
│
├── data/                        # PDF, ChromaDB, BM25 index
├── docs/                        # Architecture diagrams
│
├── .env.example                 # Environment template
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Component health check |
| POST | `/ingest` | Ingest PDF (background task) |
| GET | `/status` | Ingestion status |
| POST | `/query` | Main RAG query |
| GET | `/history` | Conversation history |
| POST | `/history/clear` | Clear history |
| GET | `/graph/chapter/{name}` | Chapter graph data |
| GET | `/stats` | Index statistics |
| GET | `/docs` | Swagger UI |

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Coulomb'\''s Law?",
    "top_k": 10,
    "use_vector": true,
    "use_bm25": true,
    "use_graph": true,
    "use_reranker": true
  }'
```

---

## 🧠 Tech Stack & Design Decisions

### Embedding Model: BAAI/bge-small-en-v1.5
- Strong MTEB benchmark performance for retrieval tasks
- Small footprint (133MB) vs BGE-large (1.2GB) with comparable accuracy
- Supports instruction-prefixed queries for asymmetric retrieval
- Free local inference, no API costs

### Vector DB: ChromaDB
- Lightweight, embeds in Python process
- Cosine similarity with HNSW indexing
- Production alternative: Qdrant or Weaviate for scale

### Keyword Search: BM25 (rank-bm25)
- Physics-aware tokenization preserves formula notation
- Handles exact terms (Coulomb, Gauss, ε₀) better than dense retrieval
- k1=1.5, b=0.75 defaults optimal for scientific text

### Graph DB: Neo4j
- Enables multi-hop reasoning: concept → law → formula
- Pre-seeded with 8-chapter physics knowledge
- Gracefully degrades if unavailable

### Reranker: BAAI/bge-reranker-base
- Cross-encoder: attends query+passage jointly → better relevance
- Input: top-20 from RRF → Output: top-5
- ~200ms overhead but significant quality improvement

### LLM: Gemini 2.5 Flash / GPT-4o
- Temperature=0.1 for factual accuracy
- Strict system prompt prevents hallucination
- Fallback response when context insufficient

### Fusion: RRF (Reciprocal Rank Fusion)
- Rank-based: immune to score-scale differences between retrievers
- k=60: empirically optimal for most retrieval tasks
- No learned parameters required

---

## 📊 Chunking Strategy

**Approach: Section-Aware + Sliding Window**

1. Parse PDF → extract pages with metadata
2. Group by chapter
3. Split on section boundaries (regex: `\d+\.\d+\s+[A-Z]`)
4. Apply sliding window within sections: 900 tokens, 175 overlap

**Tradeoffs:**

| Strategy | Pros | Cons |
|----------|------|------|
| Section-aware | Preserves context, chapter/heading metadata | Uneven chunk sizes |
| Fixed-size | Predictable, simple | May split mid-concept |
| Semantic | Best semantic boundaries | Requires embedding pass, slower |
| **Hybrid (used)** | Section boundaries + token window | More code complexity |

---

## 📈 Evaluation

Run the 50-question benchmark:

```bash
python app/evaluation/ragas_eval.py
```

**RAGAS Metrics evaluated:**
- **Faithfulness**: Are claims in the answer supported by retrieved context?
- **Answer Relevancy**: How relevant is the answer to the question?
- **Context Precision**: What fraction of retrieved context is useful?
- **Context Recall**: How much of the needed context was retrieved?

---

## ⚠️ Limitations & Improvement Roadmap

| Issue | Current Mitigation | Future Fix |
|-------|-------------------|------------|
| Formula extraction | Regex-based | Use specialized LaTeX parser / MathPix |
| Table handling | Plain text extraction | Camelot / pdfplumber for structured tables |
| Multi-page context | 175 token overlap | Increase overlap or use parent-child chunks |
| Graph quality | Seed + extraction | Fine-tune NER for physics entities |
| Hallucination | Strict system prompt | Constitutional AI + self-consistency |
| Latency | ~2-5s total | Cache embeddings, quantize reranker |
| Scale | ChromaDB in-memory | Migrate to Qdrant/Weaviate for production |

---

## 🔍 Live Demo Guide

1. Start services: `docker-compose up`
2. Open http://localhost:8501
3. Click **"🚀 Ingest PDF"** in sidebar (wait ~2-3 min)
4. Try sample questions or ask your own
5. Show the reviewer:
   - **Retrieved Chunks tab**: explainability with per-source scores
   - **Graph Retrieval tab**: Neo4j paths
   - **Retrieval Scores tab**: comparative bar chart
   - **Latency tab**: timing breakdown
6. Follow-up questions work via conversation history

---

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini 2.5 Flash API key | Required |
| `OPENAI_API_KEY` | OpenAI API key (alternative) | Optional |
| `LLM_PROVIDER` | `gemini` or `openai` | `gemini` |
| `NEO4J_URI` | Neo4j bolt URI | `bolt://localhost:7687` |
| `NEO4J_PASSWORD` | Neo4j password | `physics_rag_2024` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./data/chroma_db` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-small-en-v1.5` |
| `RERANKER_MODEL` | HuggingFace reranker model | `BAAI/bge-reranker-base` |
| `TOP_K_RETRIEVAL` | Candidates per retriever | `10` |
| `TOP_K_RERANK` | Final chunks after reranking | `5` |
| `LANGCHAIN_API_KEY` | LangSmith monitoring | Optional |
