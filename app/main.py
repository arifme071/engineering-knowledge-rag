"""
Engineering Knowledge Assistant — RAG Pipeline
Built on published research in railroad condition monitoring,
DAS signal processing, and AI-driven manufacturing optimization.

Author: Md Arifur Rahman
GitHub: https://github.com/arifme071
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.ui_helpers import render_sources, render_example_queries

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engineering Knowledge Assistant",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #1f6feb, #388bfd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-header { font-size: 1rem; color: #8b949e; margin-bottom: 1.5rem; }
.source-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.85rem;
}
.metric-pill {
    display: inline-block; background: #1a3a22; color: #56d364;
    border-radius: 20px; padding: 2px 10px; font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/RAG-Pipeline-4285F4?style=flat-square")
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2",
         "sentence-transformers/all-mpnet-base-v2"],
        help="Model used to embed documents and queries"
    )

    top_k = st.slider(
        "Retrieved chunks (top-k)", min_value=1, max_value=8, value=3,
        help="Number of document chunks to retrieve per query"
    )

    temperature = st.slider(
        "Response temperature", min_value=0.0, max_value=1.0, value=0.3,
        step=0.1, help="Lower = more focused, higher = more creative"
    )

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
    **Domain coverage:**
    - 🚂 Railroad condition monitoring
    - 📡 DAS fiber-optic signal analysis
    - 🏭 WAAM manufacturing optimization
    - 🤖 CNN-LSTM anomaly detection
    - 📊 Predictive maintenance
    """)

    st.markdown("---")
    st.markdown("### 👤 About")
    st.markdown("""
    Built by **Md Arifur Rahman**
    PIN Fellow · Georgia Tech

    [![Scholar](https://img.shields.io/badge/184%2B_Citations-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iafas1MAAAAJ)
    [![GitHub](https://img.shields.io/badge/arifme071-181717?style=flat-square&logo=github)](https://github.com/arifme071)
    """)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚂 Engineering Knowledge Assistant</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">RAG pipeline over published research in railroad AI, '
    'DAS signal processing & manufacturing optimization</div>',
    unsafe_allow_html=True
)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Papers indexed", "7")
col2.metric("Citations", "184+")
col3.metric("Domains", "3")
col4.metric("Model", "MiniLM-L6")

st.markdown("---")

# ── Initialize RAG pipeline ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline(model_name: str) -> RAGPipeline:
    pipeline = RAGPipeline(embedding_model=model_name)
    pipeline.load_or_build_index()
    return pipeline

pipeline = load_pipeline(model_choice)

# ── Example queries ───────────────────────────────────────────────────────────
st.markdown("#### 💡 Example queries")
render_example_queries()

# ── Chat interface ────────────────────────────────────────────────────────────
st.markdown("#### 🔍 Ask a question")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])

# Query input
if query := st.chat_input("e.g. How does the CNN-LSTM sliding window correct misclassifications?"):
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = pipeline.query(query, top_k=top_k, temperature=temperature)

        st.markdown(result["answer"])
        render_sources(result["sources"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

# Clear chat
if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
