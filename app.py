"""
RAG System - AI/ML Knowledge Base Chatbot
Streamlit Interface - Light Theme
"""

import json
import numpy as np
import faiss
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─────────────────────────────────────────────────
# HARDCODED API KEY — not shown in UI
# ─────────────────────────────────────────────────
GOOGLE_API_KEY = "Your-Actual-Key-Here"   # <-- paste your key here
genai.configure(api_key=GOOGLE_API_KEY)

# ─────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AI/ML Knowledge RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# LIGHT THEME CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* --- OVERALL LIGHT BACKGROUND --- */
    .main, .stApp {
        background-color: #f8fafc;
    }
    div[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }

    /* --- TITLE --- */
    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* --- HEADINGS --- */
    h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* --- METRICS --- */
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* --- TEXT AREA --- */
    .stTextArea label p {
        color: #1e293b !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px #bfdbfe !important;
    }

    /* --- CHUNK CARDS --- */
    .chunk-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .chunk-header {
        color: #3b82f6 !important;
        font-weight: 700 !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .chunk-text {
        color: #334155 !important;
        font-size: 0.9rem !important;
        line-height: 1.6;
    }

    /* --- ANSWER CARD --- */
    .answer-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 20px;
        color: #14532d !important;
        font-size: 1rem !important;
        line-height: 1.7;
    }

    /* --- MAIN BUTTON --- */
    .stButton>button {
        background: #3b82f6;
        color: white !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        width: 100%;
        padding: 0.6rem 2rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background: #2563eb !important;
        color: white !important;
    }

    /* --- SIDEBAR SAMPLE QUESTION BUTTONS --- */
    div[data-testid="stSidebar"] .stButton>button {
        background: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 0.9rem !important;
        margin-bottom: 6px !important;
        border-radius: 8px !important;
        text-align: left !important;
        width: 100% !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-testid="stSidebar"] .stButton>button:hover {
        background: #eff6ff !important;
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
    }

    /* --- SIDEBAR SECTION HEADERS --- */
    .sidebar-section-title {
        color: #1e293b;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 12px 0 8px 0;
    }

    /* --- HISTORY ITEMS --- */
    .history-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 6px;
    }
    .history-q {
        color: #475569;
        font-size: 0.8rem;
    }

    /* --- SLIDER --- */
    .stSlider label p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* --- EXPANDER --- */
    .stExpander {
        background: #ffffff;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stExpander summary p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    .stExpander .stMarkdown p,
    .stExpander .stMarkdown strong {
        color: #475569 !important;
    }
    .stExpander textarea {
        background-color: #f8fafc !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* --- DIVIDERS --- */
    hr {
        border-color: #e2e8f0 !important;
    }

    /* --- CAPTIONS --- */
    .stCaption p {
        color: #94a3b8 !important;
    }

    /* --- HIDE STREAMLIT FOOTER --- */
    footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# RESOURCE LOADERS
# ─────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_rag_system():
    index_path = Path("faiss_index.bin")
    meta_path  = Path("chunks_metadata.json")

    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r") as f:
            data = json.load(f)
        return index, data["chunks"], data["metadata"]
    return None, [], []


def retrieve_chunks(query, index, all_chunks, chunk_metadata, model, top_k=4):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        results.append({
            "rank":       rank + 1,
            "score":      float(score),
            "chunk_text": all_chunks[idx],
            "source":     chunk_metadata[idx]["source"],
            "topic":      chunk_metadata[idx]["topic"],
        })
    return results


def generate_answer(query, retrieved_chunks):
    context_parts = [
        f"[Source: {c['source']}]\n{c['chunk_text']}"
        for c in retrieved_chunks
    ]
    context = "\n\n---\n\n".join(context_parts)
    system_instruction = (
        "Answer ONLY using the provided context documents. "
        "Cite sources when possible. If the context lacks information, say so clearly. "
        "Be concise and accurate."
    )

    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(f"Context:\n{context}\n\nQuestion: {query}")
            return response.text
        except Exception:
            continue

    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            try:
                model = genai.GenerativeModel(model_name=m.name)
                response = model.generate_content(
                    f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {query}"
                )
                return response.text
            except Exception:
                continue

    return "Error: No available Gemini models found. Please check your API key and billing tier."


# ─────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────
with st.sidebar:

    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.markdown("<div class='sidebar-section-title'>Knowledge Base</div>", unsafe_allow_html=True)
    st.markdown("""
- Machine Learning
- Deep Learning
- Neural Networks
- NLP
- Large Language Models
    """)

    st.markdown("---")
    st.markdown("<div class='sidebar-section-title'>Sample Questions</div>", unsafe_allow_html=True)

    sample_questions = [
        "What is the Transformer architecture?",
        "Who invented LSTM?",
        "How does RAG reduce hallucination?",
        "What is backpropagation?",
        "Explain overfitting and how to prevent it",
        "What are CNNs used for?",
    ]
    for sq in sample_questions:
        if st.button(sq, key=f"sq_{sq}"):
            st.session_state.prefill = sq

    st.markdown("---")
    st.markdown("<div class='sidebar-section-title'>Query History</div>", unsafe_allow_html=True)
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(
                f"<div class='history-item'><div class='history-q'>"
                f"Q{len(st.session_state.history)-i}: {h['query'][:55]}..."
                f"</div></div>",
                unsafe_allow_html=True
            )
    else:
        st.caption("No queries yet")


# ── Main content ──────────────────────────────────
st.markdown("<div class='title-text'>AI/ML RAG Knowledge Base</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle-text'>Retrieval-Augmented Generation "
    "powered by FAISS + Sentence Transformers + Gemini</div>",
    unsafe_allow_html=True
)

with st.spinner("Loading RAG system..."):
    index, all_chunks, chunk_metadata = load_rag_system()
    embed_model = load_embedding_model()

if index is None:
    st.error("Missing index files! Run the notebook first to generate 'faiss_index.bin' and 'chunks_metadata.json'.")
    st.stop()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
unique_docs = len(set(m["source"] for m in chunk_metadata)) if chunk_metadata else 0
col1.metric("Documents",     str(unique_docs))
col2.metric("Chunks",        str(len(all_chunks)))
col3.metric("Embedding Dim", "384")
col4.metric("Vector DB",     "FAISS")

st.markdown("---")

# Query input
prefill = st.session_state.pop("prefill", "")
query = st.text_area(
    "Enter your question about AI/ML:",
    value=prefill,
    height=100,
    placeholder="e.g. What is the difference between supervised and unsupervised learning?"
)

search_col, _ = st.columns([1, 3])
with search_col:
    search_clicked = st.button("Search and Generate Answer")

# Results
if search_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve_chunks(
                query, index, all_chunks, chunk_metadata, embed_model, top_k=top_k
            )

        with st.spinner("Generating answer with Gemini..."):
            try:
                answer = generate_answer(query, retrieved)
            except Exception as e:
                st.error(f"Generation Error: {e}")
                st.stop()

        st.session_state.history.append({"query": query, "answer": answer})

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(f"### Retrieved Chunks (Top {top_k})")
            for r in retrieved:
                preview = r["chunk_text"][:280] + "..." if len(r["chunk_text"]) > 280 else r["chunk_text"]
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">Rank {r['rank']} | {r['source']} | Score: {r['score']:.4f}</div>
                    <div class="chunk-text">{preview}</div>
                </div>""", unsafe_allow_html=True)

        with right:
            st.markdown("### Generated Answer")
            st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)
            sources_used = list(set(r["source"] for r in retrieved))
            st.markdown("**Sources consulted:**")
            for s in sources_used:
                st.caption(f"{s}")

        with st.expander("View Full Retrieved Chunks"):
            for r in retrieved:
                st.markdown(f"**Rank {r['rank']} | {r['source']} | Score: {r['score']:.4f}**")
                st.text_area(label="", value=r["chunk_text"], height=150, key=f"full_{r['rank']}")
