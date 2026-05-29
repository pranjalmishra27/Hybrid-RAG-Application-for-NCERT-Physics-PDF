"""
Physics Hybrid RAG - Streamlit Frontend
Production-grade UI with explainability, citations, and graph visualization.
"""

import streamlit as st
import requests
import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Physics RAG | NCERT Class 12",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --text: #e2e8f0;
    --text-dim: #8892a4;
    --border: #1e2d42;
  }

  .stApp { background: var(--bg); color: var(--text); font-family: 'Sora', sans-serif; }
  .stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border); }

  h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

  .rag-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .rag-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
  }

  .answer-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
    font-size: 15px;
    line-height: 1.8;
  }

  .chunk-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    font-size: 13px;
  }
  .chunk-card:hover { border-color: var(--accent); transition: border-color 0.2s; }

  .score-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    margin: 2px;
  }
  .score-vec { background: #0d2b45; color: var(--accent); border: 1px solid var(--accent); }
  .score-bm25 { background: #1a1042; color: #a78bfa; border: 1px solid #7c3aed; }
  .score-graph { background: #0d2b1f; color: var(--accent3); border: 1px solid var(--accent3); }
  .score-rrf { background: #2b1a0d; color: #f59e0b; border: 1px solid #f59e0b; }
  .score-rerank { background: #2b0d1a; color: #f472b6; border: 1px solid #ec4899; }

  .citation-pill {
    display: inline-block;
    background: #0d2b45;
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    margin: 4px;
  }

  .latency-item {
    background: var(--surface2);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    border: 1px solid var(--border);
  }

  .query-type-badge {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 12px;
  }

  div[data-testid="stTextInput"] > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
  }
  div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #0077aa, #00d4ff) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
  }

  .status-ok { color: var(--accent3); }
  .status-err { color: #f87171; }

  .graph-path {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--accent3);
    background: #0d2b1f;
    padding: 4px 10px;
    border-radius: 4px;
    margin-top: 6px;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

def api_get(endpoint: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint: str, data: Dict, timeout: int = 120) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=timeout)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def render_score_badges(chunk: Dict):
    badges = []
    if chunk.get("vector_score", 0) > 0:
        badges.append(f'<span class="score-badge score-vec">Vec: {chunk["vector_score"]:.3f}</span>')
    if chunk.get("bm25_score", 0) > 0:
        badges.append(f'<span class="score-badge score-bm25">BM25: {chunk["bm25_score"]:.3f}</span>')
    if chunk.get("graph_score", 0) > 0:
        badges.append(f'<span class="score-badge score-graph">Graph: {chunk["graph_score"]:.2f}</span>')
    if chunk.get("rrf_score", 0) > 0:
        badges.append(f'<span class="score-badge score-rrf">RRF: {chunk["rrf_score"]:.5f}</span>')
    if chunk.get("reranker_score") is not None:
        badges.append(f'<span class="score-badge score-rerank">Rerank: {chunk["reranker_score"]:.3f}</span>')
    return " ".join(badges)


def render_chunk_card(i: int, chunk: Dict):
    meta = chunk.get("metadata", {})
    page = meta.get("page", "?")
    chapter = meta.get("chapter", "Unknown")
    subheading = meta.get("subheading", "")
    formulas = meta.get("formulas", "")
    source = chunk.get("retrieval_source", "mixed")
    graph_path = chunk.get("graph_path", "")
    content = chunk.get("content", "")[:400]

    source_colors = {"vector": "#00d4ff", "bm25": "#a78bfa", "graph": "#10b981"}
    src_color = source_colors.get(source, "#8892a4")

    html = f"""
    <div class="chunk-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <strong style="color:#e2e8f0">Chunk {i} · Page {page}</strong>
        <span style="color:{src_color}; font-size:11px; font-family:'Space Mono',monospace; 
                     background:{src_color}22; padding:2px 8px; border-radius:10px;">
          {source.upper()}
        </span>
      </div>
      <div style="color:#60a5fa; font-size:12px; margin-bottom:6px;">
        📖 {chapter}{f' › {subheading}' if subheading else ''}
      </div>
      <div style="color:#94a3b8; font-size:13px; line-height:1.6; margin-bottom:8px;">
        {content}{'...' if len(chunk.get('content','')) > 400 else ''}
      </div>
      {f'<div style="color:#fbbf24;font-size:12px;margin-bottom:6px">⚡ {formulas[:100]}</div>' if formulas else ''}
      {f'<div class="graph-path">🕸 {graph_path}</div>' if graph_path else ''}
      <div style="margin-top:8px">{render_score_badges(chunk)}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0;">
      <div style="font-size:36px">⚛️</div>
      <div style="font-family:'Space Mono',monospace; font-size:14px; color:#00d4ff; font-weight:700;">
        PHYSICS RAG
      </div>
      <div style="font-size:11px; color:#8892a4;">NCERT Class 12 · Hybrid Search</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # System Status
    st.markdown("**System Status**")
    health = api_get("/health")
    if health:
        cols = st.columns(2)
        cols[0].markdown(f"{'🟢' if health.get('vector_store') else '🔴'} Vector DB")
        cols[1].markdown(f"{'🟢' if health.get('bm25') else '🔴'} BM25")
        cols2 = st.columns(2)
        cols2[0].markdown(f"{'🟢' if health.get('graph') else '🟡'} Neo4j")
        cols2[1].markdown(f"{'🟢' if health.get('reranker') else '🔴'} Reranker")
        st.markdown(f"**LLM:** `{health.get('llm_provider', 'unknown')}`")
    else:
        st.error("API Server Unreachable")
        st.info("💡 **Backend is starting up!**\n\nThe local FastAPI server is loading heavy AI models (BGE Embeddings & Reranker) on CPU. This typically takes **20-40 seconds** on the first start. Please wait a bit and refresh below.")
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    st.divider()

    # Retrieval Settings
    st.markdown("**Retrieval Settings**")
    top_k = st.slider("Top K", min_value=3, max_value=20, value=10)
    use_reranker = st.toggle("Enable Reranker", value=True)
    use_vector = st.toggle("Vector Search", value=True)
    use_bm25 = st.toggle("BM25 Search", value=True)
    use_graph = st.toggle("Graph Search", value=True)

    st.divider()

    # Conversation
    st.markdown("**Conversation**")
    st.caption(f"History: {len(st.session_state.chat_history)} turns")
    if st.button("🗑️ Clear History"):
        api_post("/history/clear", {})
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <h1 style="margin:0; font-size:24px; color:#e2e8f0;">
    ⚛️ Physics Hybrid RAG
  </h1>
  <p style="color:#8892a4; margin:4px 0 0 0; font-size:13px;">
    NCERT Class 12 Physics Part 1 · Vector + BM25 + Graph · Grounded Answers · Citations
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sample Questions ──────────────────────────────────────────────────────────
with st.expander("💡 Sample Questions", expanded=False):
    samples = [
        "What is Coulomb's Law? State the formula.",
        "Explain Gauss's Law and its applications.",
        "What is the difference between electric field and electric potential?",
        "Derive the expression for electric field due to a point charge.",
        "What is Faraday's Law of electromagnetic induction?",
        "Explain the working principle of a transformer.",
        "What are Maxwell's equations?",
        "Define self-inductance and mutual inductance.",
    ]
    cols = st.columns(2)
    for i, q in enumerate(samples):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["prefill_query"] = q
            st.session_state["auto_submit"] = True

# ── Query Input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
auto_submit = st.session_state.pop("auto_submit", False)

with st.form("query_form", border=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Ask a Physics Question",
            value=prefill,
            placeholder="e.g. What is Coulomb's Law?",
            label_visibility="collapsed",
        )
    with col2:
        ask_btn = st.form_submit_button("Ask ⚡", use_container_width=True)

# ── Process Query ─────────────────────────────────────────────────────────────
if (ask_btn or auto_submit) and query.strip():
    with st.spinner("🔍 Retrieving from Physics knowledge base..."):
        payload = {
            "question": query,
            "top_k": top_k,
            "use_vector": use_vector,
            "use_bm25": use_bm25,
            "use_graph": use_graph,
            "use_reranker": use_reranker,
        }
        response = api_post("/query", payload, timeout=120)

    if response and "error" not in response:
        st.session_state.last_response = response
        st.session_state.chat_history.append({
            "question": query,
            "answer": response.get("answer", ""),
            "citations": response.get("citations", []),
        })
    else:
        st.error(f"Query failed: {response.get('error', 'Unknown error')}")

# ── Display Response ──────────────────────────────────────────────────────────
if st.session_state.last_response:
    resp = st.session_state.last_response
    answer = resp.get("answer", "")
    citations = resp.get("citations", [])
    chunks = resp.get("chunks", [])
    query_type = resp.get("query_type", "")
    latency = resp.get("latency", {})
    retrieval_meta = resp.get("retrieval_meta", {})

    # Answer Section
    st.markdown(f'<div class="query-type-badge">🔍 {query_type.upper().replace("_", " ")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Citations
    if citations:
        st.markdown("**📚 Sources**")
        cit_html = ""
        for c in citations:
            cit_html += f'<span class="citation-pill">📄 Page {c["page"]} · {c["chapter"]}</span>'
        st.markdown(cit_html, unsafe_allow_html=True)

    st.divider()

    # Tabs for explainability
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Retrieved Chunks",
        "🕸 Graph Retrieval",
        "📊 Retrieval Scores",
        "⏱ Latency"
    ])

    with tab1:
        st.markdown(f"**Top {len(chunks)} chunks after reranking**")
        for i, chunk in enumerate(chunks, 1):
            render_chunk_card(i, chunk)

        # Show retrieval breakdown
        vec_count = len(retrieval_meta.get("vector_results", []))
        bm25_count = len(retrieval_meta.get("bm25_results", []))
        graph_count = len(retrieval_meta.get("graph_results", []))

        st.markdown(f"""
        <div style="margin-top:16px; font-size:12px; color:#8892a4; font-family:'Space Mono',monospace;">
          Vector: {vec_count} results &nbsp;|&nbsp; BM25: {bm25_count} results &nbsp;|&nbsp; Graph: {graph_count} results
          → RRF Fusion → Reranked to top {len(chunks)}
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        graph_results = retrieval_meta.get("graph_results", [])
        if graph_results:
            st.markdown(f"**{len(graph_results)} results from knowledge graph**")
            for gr in graph_results:
                meta = gr.get("metadata", {})
                path = gr.get("graph_path", "")
                st.markdown(f"""
                <div class="chunk-card">
                  <div style="color:#60a5fa; font-size:12px; margin-bottom:4px;">
                    📖 {meta.get('chapter', 'Unknown')} · Page {meta.get('page', '?')}
                  </div>
                  {f'<div class="graph-path">🕸 {path}</div>' if path else ''}
                  <div style="color:#94a3b8; font-size:12px; margin-top:6px;">
                    {gr.get('content', '')[:200]}...
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No graph results (Neo4j may not be running or no matching nodes found)")

    with tab3:
        if chunks:
            # Score comparison chart
            df = pd.DataFrame([{
                "Chunk": f"Chunk {i+1} (p.{c.get('metadata', {}).get('page','?')})",
                "Vector": c.get("vector_score", 0),
                "BM25": c.get("bm25_score", 0),
                "Graph": c.get("graph_score", 0),
                "RRF": c.get("rrf_score", 0) * 100,
                "Reranker": c.get("reranker_score", 0) if c.get("reranker_score") is not None else 0,
            } for i, c in enumerate(chunks)])

            fig = go.Figure()
            colors = {"Vector": "#00d4ff", "BM25": "#a78bfa", "Graph": "#10b981",
                      "RRF": "#f59e0b", "Reranker": "#f472b6"}
            for col in ["Vector", "BM25", "Graph", "Reranker"]:
                fig.add_trace(go.Bar(
                    name=col, x=df["Chunk"], y=df[col],
                    marker_color=colors[col], opacity=0.85
                ))
            fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#1e2d42")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        timing = retrieval_meta.get("timing", {})
        all_timing = {**timing, "generation_ms": latency.get("generation_ms", 0)}

        t_cols = st.columns(5)
        labels = {
            "vector_ms": ("🔵", "Vector"),
            "bm25_ms": ("🟣", "BM25"),
            "graph_ms": ("🟢", "Graph"),
            "fusion_ms": ("🟡", "RRF Fusion"),
            "generation_ms": ("🔴", "LLM Gen"),
        }
        for i, (key, (icon, label)) in enumerate(labels.items()):
            val = all_timing.get(key, latency.get(key, 0))
            t_cols[i % 5].markdown(f"""
            <div class="latency-item">
              <div style="font-size:20px">{icon}</div>
              <div style="font-size:20px; font-family:'Space Mono',monospace; color:#e2e8f0;">
                {val}ms
              </div>
              <div style="font-size:11px; color:#8892a4;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        total = latency.get("total_ms", 0)
        st.markdown(f"""
        <div style="text-align:center; margin-top:16px; font-family:'Space Mono',monospace; 
                    color:#00d4ff; font-size:18px;">
          Total: {total}ms
        </div>
        """, unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────────
if len(st.session_state.chat_history) > 1:
    st.divider()
    with st.expander(f"💬 Conversation History ({len(st.session_state.chat_history)} turns)", expanded=False):
        for i, turn in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {turn['question']}")
            st.markdown(f"**A:** {turn['answer'][:300]}...")
            if turn.get("citations"):
                cits = ", ".join([f"p.{c['page']}" for c in turn["citations"][:3]])
                st.caption(f"📄 Sources: {cits}")
            st.divider()
