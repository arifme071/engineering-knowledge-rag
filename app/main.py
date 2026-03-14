"""
Engineering Knowledge Assistant — RAG Pipeline
Author: Md Arifur Rahman | github.com/arifme071
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Engineering Knowledge Assistant",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title { font-size:2rem; font-weight:700; color:#1f6feb; margin-bottom:0.2rem; }
.sub-title  { font-size:1rem; color:#8b949e; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Retrieved chunks (top-k)", 1, 6, 3)
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
**Domains:**
- 🚂 Railroad condition monitoring
- 📡 DAS fiber-optic signal analysis
- 🏭 WAAM manufacturing (HMM-RL)
- 🤖 CNN-LSTM anomaly detection
    """)
    st.markdown("---")
    st.markdown("""
**Md Arifur Rahman** · PIN Fellow, Georgia Tech

[![Scholar](https://img.shields.io/badge/184%2B_Citations-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iafas1MAAAAJ)
[![GitHub](https://img.shields.io/badge/arifme071-181717?style=flat-square&logo=github)](https://github.com/arifme071)
    """)

st.markdown('<div class="main-title">🚂 Engineering Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">RAG pipeline over 7 published papers in railroad AI, DAS signal processing & manufacturing optimization</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Papers indexed", "7")
c2.metric("Citations", "184+")
c3.metric("Embedding model", "MiniLM-L6")
c4.metric("Vector store", "FAISS")
st.markdown("---")

KNOWLEDGE = [
    {"title":"CNN-LSTM-SW for Railroad Anomaly Detection","authors":"Rahman MA, Jamal S, Taheri H","year":"2024","venue":"Elsevier GEITS","doi":"https://doi.org/10.1016/j.geits.2024.100178","domain":"railroad","text":"The CNN-LSTM-SW model combines CNNs for spatial feature extraction, LSTMs for temporal pattern learning, and a sliding window (SW) post-processing step correcting point-to-point misclassifications. Validated on HTL loop DAS data from AAR/TTCI facility in Pueblo CO, covering 4.16 km of fiber-optic track with 310,000 samples."},
    {"title":"CNN-LSTM-SW — Sliding Window Correction","authors":"Rahman MA, Jamal S, Taheri H","year":"2024","venue":"Elsevier GEITS","doi":"https://doi.org/10.1016/j.geits.2024.100178","domain":"railroad","text":"The SW corrects CNN-LSTM boundary misclassifications between TP and AC2. A majority-vote window considers neighboring predictions and assigns the majority class above a threshold, reducing isolated errors while preserving true anomaly boundaries."},
    {"title":"CNN-LSTM-SW — Condition Classes and Results","authors":"Rahman MA, Jamal S, Taheri H","year":"2024","venue":"Elsevier GEITS","doi":"https://doi.org/10.1016/j.geits.2024.100178","domain":"railroad","text":"Four classes: NC (Normal ~93%), TP (Train Position), AC1 (Anomaly Class 1 - wheel flat), AC2 (Anomaly Class 2 - rail joint). SMOTE handles class imbalance. ROC AUC: AC1=1.00, NC=0.99, TP=0.97, AC2=0.95. Best model achieved 97% train position detection using combined T&FD features."},
    {"title":"CNN-LSTM-SW — Feature Extraction","authors":"Rahman MA, Jamal S, Taheri H","year":"2024","venue":"Elsevier GEITS","doi":"https://doi.org/10.1016/j.geits.2024.100178","domain":"railroad","text":"Three feature domains from DAS signals: Time domain (mean, Pearson skewness, kurtosis), Frequency domain (band energy ratio, spectral kurtosis, spectral contrast via FFT), Time-frequency domain (STFT spectral mean/std/max/median). Combined T&FD (97%) outperforms TD-only (89%) and FD-only (88%)."},
    {"title":"DAS Railroad Monitoring with GRU/LSTM","authors":"Rahman MA, Kim J, Dababneh F, Taheri H","year":"2024","venue":"SPIE JARS","doi":"https://doi.org/10.1117/1.JRS.18.016512","domain":"das","text":"GRU achieved 94% vs LSTM 93% for train presence detection. GRU preferred for DAS due to fewer parameters and faster convergence. Average trend extraction preprocesses noisy DAS via rolling mean across multi-channel matrix, smoothing transient noise while retaining sustained acoustic events."},
    {"title":"How Distributed Acoustic Sensing Works","authors":"Rahman MA, Kim J, Dababneh F, Taheri H","year":"2024","venue":"SPIE JARS","doi":"https://doi.org/10.1117/1.JRS.18.016512","domain":"das","text":"DAS uses fiber optic cables as distributed sensors via Rayleigh backscattering. iDAS system (Silixa): gauge length 10m, sensing range 45km, sampling up to 100kHz. Advantages: distributed sensing over km from one interrogator, no electrical power along fiber, immunity to EMI, simultaneous monitoring of multiple events."},
    {"title":"Review: DAS Applications for Railroad CM","authors":"Rahman MA et al.","year":"2023","venue":"Mechanical Systems and Signal Processing, Elsevier","doi":"https://doi.org/10.1016/j.ymssp.2023.110881","domain":"das","text":"DAS challenges: terabytes of data daily, high environmental noise, class imbalance, real-time processing constraints. Deep learning (CNN, LSTM, GRU hybrids) consistently outperforms classical ML (SVM, KNN, Random Forest) on large DAS datasets through automatic multi-scale temporal and spectral feature learning."},
    {"title":"HMM-RL for WAAM Manufacturing Optimization","authors":"Rahman MA et al.","year":"2026","venue":"Springer","doi":"Under review","domain":"manufacturing","text":"HMM-RL pipeline for Wire Arc Additive Manufacturing: HMM models latent process states from sensor observations (stable deposition, thermal buildup, cooling), RL optimizes control policy across states. Adjusts wire feed speed, travel speed, current, and gas flow to maximize material quality. Improved material utilization by 5% on live WAAM datasets under Georgia-AIM grant at Georgia Tech."},
]

@st.cache_resource(show_spinner="🔄 Loading knowledge base (first load ~30s)...")
def build_index():
    from sentence_transformers import SentenceTransformer
    import faiss
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [k["text"] for k in KNOWLEDGE]
    embs = model.encode(texts, normalize_embeddings=True).astype("float32")
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return model, idx

model, index = build_index()

def retrieve(query, top_k=3):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, top_k)
    return [dict(KNOWLEDGE[i], score=float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]

def answer(query, sources):
    if not sources:
        return "No relevant passages found."
    top = sources[0]
    out = f"**From: {top['title']} ({top['year']})**\n\n{top['text']}\n\n"
    if len(sources) > 1:
        out += "**Additional context:**\n"
        for s in sources[1:]:
            out += f"- *{s['title']}*: {s['text'][:180]}...\n"
    out += f"\n---\n*{len(sources)} passage(s) retrieved · Sources shown below*"
    return out

# Example queries
st.markdown("#### 💡 Try an example")
EXAMPLES = [
    "How does the CNN-LSTM sliding window correct misclassifications?",
    "What are the four condition classes in the DAS dataset?",
    "How does SMOTE handle class imbalance in training?",
    "What is the difference between GRU and LSTM for DAS?",
    "How does HMM-RL optimize WAAM manufacturing?",
    "What features are extracted from DAS signals?",
]
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(f"💬 {ex[:52]}...", key=f"ex{i}", use_container_width=True):
        st.session_state["prefill"] = ex

st.markdown("#### 🔍 Ask a question")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s)", expanded=False):
                for i, s in enumerate(msg["sources"]):
                    st.markdown(f"**[{i+1}] {s['title']}** · {s['authors']} · {s['venue']} {s['year']} · relevance: {int(s['score']*100)}%\n\n{s['text']}")
                    if s['doi'].startswith('http'):
                        st.markdown(f"[📄 Paper link]({s['doi']})")
                    st.markdown("---")

if query := st.chat_input("Ask about railroad AI, DAS sensing, or WAAM manufacturing..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            sources = retrieve(query, top_k=top_k)
            ans = answer(query, sources)
        st.markdown(ans)
        if sources:
            with st.expander(f"📚 {len(sources)} source(s)", expanded=False):
                for i, s in enumerate(sources):
                    st.markdown(f"**[{i+1}] {s['title']}** · {s['authors']} · {s['venue']} {s['year']} · relevance: {int(s['score']*100)}%\n\n{s['text']}")
                    if s['doi'].startswith('http'):
                        st.markdown(f"[📄 Paper link]({s['doi']})")
                    st.markdown("---")
    st.session_state.messages.append({"role":"assistant","content":ans,"sources":sources})

if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
