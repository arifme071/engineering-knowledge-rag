"""
Engineering Knowledge Assistant — RAG Pipeline
Comprehensive knowledge base: research papers, work experience,
education, certifications, and technical skills.
Author: Md Arifur Rahman | github.com/arifme071
"""

import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Engineering Knowledge Assistant",
    page_icon="🚂", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title { font-size:2rem; font-weight:700; color:#1f6feb; margin-bottom:0.2rem; }
.sub-title  { font-size:1rem; color:#8b949e; margin-bottom:1.5rem; }
.cat-badge  {
    display:inline-block; border-radius:20px; padding:2px 10px;
    font-size:0.72rem; font-weight:600; margin:2px;
}
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {
    "research":      ("#dafbe1", "#116329"),
    "experience":    ("#ddf4ff", "#0550ae"),
    "education":     ("#fff8c5", "#7d4e00"),
    "certifications":("#ffd8d3", "#82071e"),
    "skills":        ("#f1f0ff", "#4c2889"),
    "profile":       ("#f6f8fa", "#24292f"),
}

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Retrieved chunks (top-k)", 1, 8, 3)

    st.markdown("---")
    st.markdown("### 🔎 Filter by category")
    categories = ["All", "Research Papers", "Work Experience",
                  "Education", "Certifications", "Skills & Tools", "Profile"]
    cat_filter = st.selectbox("Category", categories)

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
**Covers:**
- 📄 All 7 published papers
- 💼 Work experience 2014–present
- 🎓 Education (CUET → GSU → Georgia Tech)
- 🏅 Certifications & professional dev
- 🛠 Technical skills & tools
    """)
    st.markdown("---")
    st.markdown("""
**Md Arifur Rahman**
PIN Fellow · Georgia Tech

[![Scholar](https://img.shields.io/badge/184%2B_Citations-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iafas1MAAAAJ)
[![GitHub](https://img.shields.io/badge/arifme071-181717?style=flat-square&logo=github)](https://github.com/arifme071)
    """)

st.markdown('<div class="main-title">🚂 Engineering Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comprehensive RAG pipeline — research papers, work experience, education, certifications & skills</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Papers indexed", "7")
c2.metric("Citations", "184+")
c3.metric("Experience", "10+ yrs")
c4.metric("Knowledge chunks", "18")
st.markdown("---")

# ── Knowledge base ────────────────────────────────────────────────────────────
KNOWLEDGE = [
    # Profile
    {"title":"Professional Profile","category":"profile","text":"Md Arifur Rahman is an ML Engineer and AI Researcher in Atlanta GA, authorized to work in the US with no sponsorship required. He has strong foundations in analytics, software development, ML, and data science. He applies advanced research to enterprise use cases — generative AI, predictive maintenance, real-time dashboards, and cloud-native ML pipelines. Contact: arifme071@gmail.com | 912-541-9169 | linkedin.com/in/marahman-gsu | github.com/arifme071"},
    {"title":"Target Roles & Career Goals","category":"profile","text":"Arifur is open to ML Engineering, AI/Data Science, Data Engineering, and AI Research roles — especially in manufacturing, transportation, energy, and infrastructure. Target companies: Google, Microsoft, Intel, and major tech firms. Unique combination: deep domain expertise (railroad AI, manufacturing) + production ML skills (RAG, CNN-LSTM, MLOps). US work authorized, no sponsorship needed. Based in Atlanta GA. Starting Georgia Tech OMSCS Fall 2026."},
    # Experience
    {"title":"Current: PIN Fellow — AI in Manufacturing, Georgia Tech (Jul 2025–Present)","category":"experience","text":"PIN Fellow at Partnership for Innovation (Georgia Tech) under the Georgia-AIM grant. Develops ML prototypes — HMM and RL pipelines on WAAM datasets. Engineers AI-driven tools for production alignment with demand/supply signals. Establishes MLOps best practices. Achievements: 5% material utilization improvement via HMM-RL on WAAM; deployed AI-driven material design prototype; authored Springer 2026 publication."},
    {"title":"Previous: Data Analyst, Norfolk Southern Corporation (Jan 2024–Mar 2025)","category":"experience","text":"Data Analyst/Reporting Supervisor Associate on the Digital Train Inspection team at Norfolk Southern, Atlanta GA. Built GIS-enabled Power BI/Tableau dashboards for train health monitoring adopted company-wide. Automated Alteryx/SQL workflows reducing manual processing 3%. Identified inspection inefficiencies improving turnaround 5%. Delivered end-to-end solutions from server-side processing to automated stakeholder reporting."},
    {"title":"Research Assistant — ML & Data Analytics, Georgia Southern University (Aug 2022–Dec 2023)","category":"experience","text":"Research Assistant at LANDTIE Lab, Georgia Southern University, Statesboro GA. Built ML anomaly detection models on live HTL loop-MxV rail datasets (AAR/TTCI, Pueblo CO) achieving 95% accuracy. Published 7 peer-reviewed papers in ASME, Springer, SPIE, Elsevier — 184+ citations. Developed Python/SQL dashboards for real-time asset monitoring. Led data collection under AAR/TTCI and Georgia Southern University grants."},
    {"title":"Deputy Manager, Aramit Limited, Bangladesh (Feb 2018–Jul 2022)","category":"experience","text":"Deputy Manager at Aramit Limited Bangladesh for 4.5 years. Led mechanical and process development, oversaw equipment maintenance, managed preventive maintenance initiatives. Expertise in hydraulic/pneumatic systems, production management, cross-functional engineering teams. Provided industrial manufacturing foundation that informs current AI/ML research."},
    {"title":"Assistant Engineer, KSRM Billet Industries, Bangladesh (2014–2018)","category":"experience","text":"First professional role as Assistant Engineer at KSRM Billet Industries Ltd, Bangladesh steel manufacturer. Gained 4 years of mechanical engineering, equipment maintenance, and industrial process management experience. Total professional experience: 10+ years spanning manufacturing operations (Bangladesh 2014–2022) and data analytics/AI research/ML engineering (USA 2022–present)."},
    # Education
    {"title":"Admitted: Georgia Tech OMSCS — MS Computer Science, Fall 2026","category":"education","text":"Admitted to Georgia Institute of Technology Online MS in Computer Science (OMSCS) starting Fall 2026, part-time online. Georgia Tech OMSCS is ranked among the best value CS master's globally. Reflects strong academic and research foundation built through 7 peer-reviewed publications and 184+ citations."},
    {"title":"MSc Applied Engineering (Advanced Manufacturing), Georgia Southern University","category":"education","text":"MSc in Applied Engineering with Advanced Manufacturing Engineering specialization from Georgia Southern University, Statesboro GA. Graduate Research Assistant at LANDTIE Research Lab under Prof. Hossein Taheri. Thesis: acoustic sensing for railroad predictive maintenance using deep learning. Gene Haas Foundation Manufacturing Engineering Scholarship 2022–2023. Produced 7 publications and 184+ citations during this degree."},
    {"title":"BSc Mechanical Engineering, CUET Bangladesh (2014)","category":"education","text":"Bachelor of Science in Mechanical Engineering from Chittagong University of Engineering and Technology (CUET), Bangladesh, graduated 2014. Merit position 35 out of 127 students. CUET is Bangladesh's top engineering university. Foundation in mechanical systems, thermodynamics, materials science, and manufacturing processes underpins later AI/manufacturing research."},
    # Research papers
    {"title":"Paper 1: CNN-LSTM-SW Railroad Anomaly Detection (Elsevier GEITS 2024)","category":"research","text":"DOI: 10.1016/j.geits.2024.100178. Rahman MA, Jamal S, Taheri H. Hybrid CNN-LSTM with sliding window post-processing for railroad anomaly detection via DAS. 4.16 km HTL loop, 310K samples, AAR/TTCI Pueblo CO. Four classes: NC, TP, AC1, AC2. Results: 97% train position detection. ROC AUC: AC1=1.00, NC=0.99, TP=0.97, AC2=0.95. SMOTE for class imbalance. T&FD features (97%) beat TD-only (89%) and FD-only (88%)."},
    {"title":"Paper 2: DAS Railroad CM with GRU/LSTM (SPIE JARS 2024)","category":"research","text":"DOI: 10.1117/1.JRS.18.016512. Rahman MA, Kim J, Dababneh F, Taheri H. GRU achieved 94% vs LSTM 93% for train detection. iDAS (Silixa): 10m gauge, 45km range, 100kHz sampling. DAS uses Rayleigh backscattering fiber optic sensing. Average trend extraction smooths noisy signals. GRU preferred for fewer parameters and faster convergence on time-series data."},
    {"title":"Paper 3: Review of DAS for Railroad CM (Elsevier MSSP 2023)","category":"research","text":"DOI: 10.1016/j.ymssp.2023.110881. Systematic review of DAS applications for railroad condition monitoring. Challenges: TB-scale daily data, environmental noise, class imbalance, real-time constraints. Deep learning (CNN/LSTM/GRU hybrids) outperforms classical ML (SVM, KNN, Random Forest) through automatic multi-scale feature learning. Widely cited review paper."},
    {"title":"Paper 4: HMM-RL for WAAM Manufacturing (Springer 2026)","category":"research","text":"Under review Springer 2026. HMM-RL for Wire Arc Additive Manufacturing intelligent control. HMM models process states (stable deposition, thermal buildup, cooling) from sensors. RL agent optimizes wire feed speed, travel speed, current, shielding gas flow. Achieved 5% material utilization improvement on live WAAM data under Georgia-AIM grant, Georgia Tech."},
    {"title":"Paper 5: AI-Guided Polymer Film Synthesis Optimization (Springer 2026)","category":"research","text":"Under review Springer 2026. AI-guided optimization of polymer film synthesis using ML surrogates and multi-objective design. ML surrogates replace expensive physical simulations. Multi-objective optimization balances competing material properties (strength, flexibility, thermal stability). Part of Georgia-AIM manufacturing AI initiative at Georgia Tech."},
    {"title":"Papers 6-7: Structural Health Monitoring & Predictive Maintenance (ASME/Springer)","category":"research","text":"Two additional publications in ASME and Springer journals covering: (1) Structural health monitoring for railroad infrastructure using sensor data and ML for early detection of structural degradation. (2) Predictive maintenance for rail systems using ML achieving 95% accuracy on HTL loop-MxV datasets with reduced false alarms. Total: 7 peer-reviewed papers, 184+ citations across ASME, Springer, SPIE (JARS), Elsevier (GEITS, MSSP)."},
    # Certifications
    {"title":"Certifications & Professional Development","category":"certifications","text":"Formal certifications: Google Cloud Data Analytics Certificate; Alteryx Designer Core Certification; Coursera Applied Machine Learning in Python (University of Michigan). Professional development: Oracle SQL Explorer; Udemy LLM Engineering; Udemy Google Cloud Professional Data Engineer prep; LinkedIn Learning Power BI; Udemy ML for Industry 4.0; Udemy Data Analyst Bootcamp; Google Generative AI Leader program."},
    # Skills
    {"title":"Technical Skills — Programming, ML & AI","category":"skills","text":"Languages: Python, SQL, Java, MATLAB, C++. ML/AI: TensorFlow, Hugging Face, LLMs (GPT/BERT), Vertex AI, RAG pipelines, AutoML, LangChain, LangGraph, FAISS, SentenceTransformers. ML methods: CNN, LSTM, GRU, HMM, Reinforcement Learning, Anomaly Detection, Predictive Maintenance, Time-Series Analysis, Signal Processing (FFT/STFT). Data: Pandas, scikit-learn, NumPy, SciPy, Alteryx. Engineering: Git, REST APIs, Agile, Streamlit, Docker."},
    {"title":"Technical Skills — Cloud, Visualization & Engineering Software","category":"skills","text":"Cloud: Google Cloud (BigQuery, Vertex AI, Cloud Functions), AWS (S3, Lambda), Azure (VM), Oracle. Visualization: Power BI, Tableau, Looker, Streamlit, GIS mapping. Engineering tools: AutoCAD, SolidWorks, Ansys simulation, PLC diagnostics and troubleshooting. Domain expertise: Manufacturing & Transportation Optimization, Asset & Structural Health Monitoring, MLOps & Model Deployment, Data Pipeline Automation, Generative AI & LLMs."},
]

CAT_MAP = {
    "Research Papers": "research",
    "Work Experience": "experience",
    "Education": "education",
    "Certifications": "certifications",
    "Skills & Tools": "skills",
    "Profile": "profile",
    "All": None,
}

@st.cache_resource(show_spinner="🔄 Loading knowledge base (~30s first load)...")
def build_index():
    from sentence_transformers import SentenceTransformer
    import faiss, numpy as np
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [k["text"] for k in KNOWLEDGE]
    embs = model.encode(texts, normalize_embeddings=True).astype("float32")
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return model, idx

model, index = build_index()

def retrieve(query, top_k=3, cat=None):
    import numpy as np
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, len(KNOWLEDGE))
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0: continue
        item = KNOWLEDGE[i]
        if cat and item["category"] != cat: continue
        results.append(dict(item, score=float(score)))
        if len(results) >= top_k: break
    return results

def build_answer(query, sources):
    if not sources: return "No relevant information found in the knowledge base."
    top = sources[0]
    out = f"**From: {top['title']}**\n\n{top['text']}\n\n"
    if len(sources) > 1:
        out += "**Additional context:**\n"
        for s in sources[1:]:
            out += f"- *{s['title']}*: {s['text'][:200]}...\n"
    out += f"\n---\n*{len(sources)} passage(s) retrieved · Expand below for sources*"
    return out

def cat_badge(cat):
    bg, fg = CATEGORY_COLORS.get(cat, ("#eee", "#333"))
    return f'<span class="cat-badge" style="background:{bg};color:{fg}">{cat}</span>'

# ── Example queries ───────────────────────────────────────────────────────────
st.markdown("#### 💡 Try an example")
EXAMPLES = [
    ("📄 Research", "How does the CNN-LSTM sliding window correct misclassifications?"),
    ("📄 Research", "What ROC AUC scores did the model achieve on each class?"),
    ("💼 Experience", "What did Arifur do at Norfolk Southern?"),
    ("💼 Experience", "What is his experience in manufacturing?"),
    ("🎓 Education", "What is Arifur's educational background?"),
    ("🏅 Certs", "What certifications does Arifur have?"),
    ("🛠 Skills", "What cloud platforms and ML tools does he use?"),
    ("👤 Profile", "Is Arifur authorized to work in the US?"),
]
cols = st.columns(2)
for i, (tag, ex) in enumerate(EXAMPLES):
    if cols[i % 2].button(f"{tag} {ex[:45]}...", key=f"ex{i}", use_container_width=True):
        st.session_state["prefill"] = ex

# ── Chat ──────────────────────────────────────────────────────────────────────
st.markdown("#### 🔍 Ask anything")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s)", expanded=False):
                for i, s in enumerate(msg["sources"]):
                    bg, fg = CATEGORY_COLORS.get(s["category"], ("#eee","#333"))
                    st.markdown(
                        f"**[{i+1}] {s['title']}**  "
                        f"<span style='background:{bg};color:{fg};border-radius:20px;"
                        f"padding:1px 8px;font-size:0.75rem'>{s['category']}</span>  "
                        f"relevance: **{int(s['score']*100)}%**\n\n{s['text']}",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

active_cat = CAT_MAP.get(cat_filter)
placeholder = "Ask about research, experience, education, skills, or certifications..."

if query := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            sources = retrieve(query, top_k=top_k, cat=active_cat)
            answer = build_answer(query, sources)
        st.markdown(answer)
        if sources:
            with st.expander(f"📚 {len(sources)} source(s)", expanded=False):
                for i, s in enumerate(sources):
                    bg, fg = CATEGORY_COLORS.get(s["category"], ("#eee","#333"))
                    st.markdown(
                        f"**[{i+1}] {s['title']}**  "
                        f"<span style='background:{bg};color:{fg};border-radius:20px;"
                        f"padding:1px 8px;font-size:0.75rem'>{s['category']}</span>  "
                        f"relevance: **{int(s['score']*100)}%**\n\n{s['text']}",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})

if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
