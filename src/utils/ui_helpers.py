"""UI helper components for the Streamlit app."""

import streamlit as st


EXAMPLE_QUERIES = [
    "How does the CNN-LSTM sliding window correct misclassifications?",
    "What are the four condition classes in the DAS railroad dataset?",
    "How does SMOTE handle class imbalance in the training data?",
    "What is the difference between GRU and LSTM for DAS signal processing?",
    "How does the HMM-RL pipeline optimize WAAM manufacturing?",
    "What features are extracted from DAS signals for model training?",
    "What is the ROC AUC score for each condition class?",
    "How does distributed acoustic sensing work on railroad tracks?",
]


def render_example_queries():
    """Render clickable example query buttons."""
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES[:6]):
        with cols[i % 2]:
            if st.button(f"💬 {q[:55]}...", key=f"ex_{i}",
                         use_container_width=True):
                st.session_state["prefill_query"] = q


def render_sources(sources: list[dict]):
    """Render source cards below an answer."""
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} source(s) retrieved", expanded=False):
        for i, src in enumerate(sources):
            score_pct = int(src.get("score", 0) * 100)
            st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
     padding:12px 16px;margin:6px 0;font-size:0.85rem;">
  <strong style="color:#58a6ff">[{i+1}] {src.get('title', 'Unknown')}</strong><br>
  <span style="color:#8b949e">{src.get('authors', '')} · {src.get('year', '')} · {src.get('venue', '')}</span><br>
  <span style="background:#1a3a22;color:#56d364;border-radius:20px;
       padding:1px 8px;font-size:0.75rem;">relevance: {score_pct}%</span>
  <span style="background:#0d2a4a;color:#79c0ff;border-radius:20px;
       padding:1px 8px;font-size:0.75rem;margin-left:4px;">{src.get('domain', '').replace('_', ' ')}</span>
  <p style="margin-top:8px;color:#c9d1d9;line-height:1.5">{src.get('text', '')[:300]}...</p>
</div>
""", unsafe_allow_html=True)
