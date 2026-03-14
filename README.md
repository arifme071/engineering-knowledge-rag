# Engineering Knowledge Assistant — RAG Pipeline

[![Live Demo](https://img.shields.io/badge/Live_Demo-HuggingFace_Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/arifme071/engineering-knowledge-rag)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-4285F4?style=flat-square)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> A production-grade Retrieval-Augmented Generation (RAG) pipeline grounded in published
> research on railroad condition monitoring, DAS signal processing, and AI-driven
> manufacturing optimization.

**🔗 Try it live →** [huggingface.co/spaces/arifme071/engineering-knowledge-rag](https://huggingface.co/spaces/arifme071/engineering-knowledge-rag)

---

## What it does

This app lets you query a knowledge base built from peer-reviewed research papers using
natural language. Type a question, and the pipeline:

1. **Retrieves** the most semantically relevant passages from the knowledge base (FAISS + SentenceTransformers)
2. **Augments** a prompt with those passages as context
3. **Generates** a grounded, citation-backed answer (HuggingFace Flan-T5)

No hallucination about domain specifics — every answer is anchored to real research.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   SentenceTransformers  │  → Embed query (MiniLM-L6-v2, 384-dim)
│   (all-MiniLM-L6-v2)   │
└────────────┬────────────┘
             │ query vector
             ▼
┌─────────────────────────┐
│    FAISS Vector Index   │  → Cosine similarity search (Inner Product)
│    (IndexFlatIP)        │    Returns top-k most relevant chunks
└────────────┬────────────┘
             │ retrieved chunks + scores
             ▼
┌─────────────────────────┐
│    Prompt Builder       │  → Assembles context + question into LLM prompt
│                         │    with source attribution
└────────────┬────────────┘
             │ prompt
             ▼
┌─────────────────────────┐
│  HuggingFace Flan-T5    │  → Generates grounded answer
│  (google/flan-t5-base)  │    Open-source, runs locally — no API key needed
└────────────┬────────────┘
             │ answer + sources
             ▼
┌─────────────────────────┐
│   Streamlit Chat UI     │  → Displays answer with expandable source cards
│                         │    showing relevance scores and paper metadata
└─────────────────────────┘
```

---

## Knowledge Base

The app ships with a built-in knowledge base from **7 peer-reviewed publications** (184+ citations):

| Paper | Venue | Year | Domain |
|---|---|---|---|
| [CNN-LSTM-SW for Railroad Anomaly Detection via DAS](https://doi.org/10.1016/j.geits.2024.100178) | Elsevier GEITS | 2024 | Railroad AI |
| [DAS-based Railroad CM with GRU/LSTM](https://doi.org/10.1117/1.JRS.18.016512) | SPIE JARS | 2024 | DAS Signal Processing |
| [Review of DAS Applications for Railroad CM](https://www.sciencedirect.com/science/article/abs/pii/S0888327023008919) | Elsevier MSSP | 2023 | DAS Review |
| HMM-RL for WAAM Intelligent Control | Springer | 2026 | Manufacturing AI |

**You can extend the knowledge base** by dropping your own PDFs into `data/papers/` — the pipeline auto-ingests and indexes them on first run.

---

## Example queries

```
How does the CNN-LSTM sliding window correct misclassifications?
What are the four condition classes in the DAS railroad dataset?
How does SMOTE handle class imbalance in the training data?
What is the difference between GRU and LSTM for DAS signal processing?
How does the HMM-RL pipeline optimize WAAM manufacturing?
What features are extracted from DAS signals for model training?
What is the ROC AUC score for each condition class?
How does distributed acoustic sensing work on railroad tracks?
```

---

## Repository Structure

```
engineering-knowledge-rag/
├── app/
│   └── main.py                        # Streamlit UI — chat interface, settings sidebar
│
├── src/
│   ├── retrieval/
│   │   └── rag_pipeline.py            # Core RAG: embed → FAISS search → LLM generate
│   ├── ingestion/
│   │   ├── builtin_knowledge.py       # Pre-loaded paper excerpts (works offline)
│   │   └── document_loader.py         # PDF/TXT ingestion + chunking pipeline
│   └── utils/
│       └── ui_helpers.py              # Source cards, example query buttons
│
├── data/
│   ├── papers/                        # Drop your PDFs here to extend knowledge base
│   └── index/                         # Auto-generated FAISS index (git-ignored)
│
├── Dockerfile                         # Container for HuggingFace Spaces deployment
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### Run locally

```bash
git clone https://github.com/arifme071/engineering-knowledge-rag.git
cd engineering-knowledge-rag
pip install -r requirements.txt
streamlit run app/main.py
```

Open `http://localhost:8501` — loads instantly with the built-in knowledge base.

### Add your own documents

```bash
# Drop PDFs into data/papers/
cp your_paper.pdf data/papers/

# Delete cached index so it rebuilds
rm -rf data/index/

# Rerun — auto-ingests on startup
streamlit run app/main.py
```

### Run with Docker

```bash
docker build -t engineering-rag .
docker run -p 7860:7860 engineering-rag
```

---

## Deploy to HuggingFace Spaces (free)

1. Go to [huggingface.co](https://huggingface.co) → **New Space**
2. Name: `engineering-knowledge-rag` | SDK: **Streamlit** | Visibility: **Public**
3. Upload all files from this repo
4. HuggingFace builds and deploys automatically (~5 min)
5. Live URL: `https://huggingface.co/spaces/YOUR_USERNAME/engineering-knowledge-rag`

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, high-quality 384-dim, runs on CPU |
| **Vector store** | FAISS `IndexFlatIP` | Exact cosine similarity, no server needed |
| **LLM** | `google/flan-t5-base` | Open-source, instruction-tuned, zero API cost |
| **Frontend** | Streamlit | Shareable demo, native chat components |
| **Deployment** | HuggingFace Spaces + Docker | Free hosting, permanent public URL |

---

## Extending the Pipeline

**Swap the LLM** — any HuggingFace model works:
```python
llm_model: str = "mistralai/Mistral-7B-Instruct-v0.1"  # Better quality, needs GPU
llm_model: str = "google/flan-t5-large"                 # Bigger, still CPU-friendly
```

**Use OpenAI/Anthropic instead:**
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

**Persistent vector store with ChromaDB:**
```python
import chromadb
client = chromadb.PersistentClient(path="data/chroma")
```

---

## Related Repositories

- [railroad-anomaly-detection-cnn-lstm](https://github.com/arifme071/railroad-anomaly-detection-cnn-lstm) — CNN-LSTM-SW paper repo (Elsevier GEITS 2024)
- 📚 [Google Scholar](https://scholar.google.com/citations?user=iafas1MAAAAJ&hl=en) — Full publication list

---

## Author

**Md Arifur Rahman**
PIN Fellow (AI in Manufacturing) · Georgia Tech | MSc Applied Engineering · Georgia Southern University

[![Google Scholar](https://img.shields.io/badge/Google_Scholar-184%2B_Citations-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iafas1MAAAAJ&hl=en)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-marahman--gsu-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marahman-gsu/)
[![GitHub](https://img.shields.io/badge/GitHub-arifme071-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/arifme071)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
