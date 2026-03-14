"""
RAG Pipeline — Retrieval-Augmented Generation
Uses FAISS vector store + HuggingFace embeddings + local LLM for generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Lazy imports — only loaded when needed to keep startup fast
_faiss = None
_SentenceTransformer = None
_pipeline_hf = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_encoder(model_name: str):
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer(model_name)


class DocumentChunk:
    """A single chunk of text from the knowledge base."""

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata  # {source, title, authors, year, domain}

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentChunk":
        return cls(d["text"], d["metadata"])


class RAGPipeline:
    """
    Full RAG pipeline:
      1. Document ingestion + chunking
      2. Embedding with SentenceTransformers
      3. FAISS vector index for semantic search
      4. HuggingFace LLM for answer generation
    """

    INDEX_PATH = Path("data/index/faiss.index")
    CHUNKS_PATH = Path("data/index/chunks.json")
    DOCS_PATH = Path("data/papers")

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-base",
    ):
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.encoder = None
        self.index = None
        self.chunks: list[DocumentChunk] = []
        self._llm = None

    # ── Encoder ──────────────────────────────────────────────────────────────

    def _load_encoder(self):
        if self.encoder is None:
            self.encoder = _get_encoder(self.embedding_model_name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._load_encoder()
        return self.encoder.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=False)

    # ── Index management ─────────────────────────────────────────────────────

    def load_or_build_index(self):
        """Load existing FAISS index or build from documents."""
        if self.INDEX_PATH.exists() and self.CHUNKS_PATH.exists():
            self._load_index()
        else:
            self._build_index()

    def _load_index(self):
        faiss = _get_faiss()
        self.index = faiss.read_index(str(self.INDEX_PATH))
        with open(self.CHUNKS_PATH) as f:
            self.chunks = [DocumentChunk.from_dict(d) for d in json.load(f)]

    def _build_index(self):
        """Ingest documents, embed, and build FAISS index."""
        from src.ingestion.document_loader import load_all_documents

        self.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load and chunk documents
        self.chunks = load_all_documents(self.DOCS_PATH)

        if not self.chunks:
            # Fall back to built-in knowledge base
            from src.ingestion.builtin_knowledge import get_builtin_chunks
            self.chunks = get_builtin_chunks()

        # Embed all chunks
        texts = [c.text for c in self.chunks]
        embeddings = self._embed(texts).astype("float32")

        # Build FAISS flat L2 index
        faiss = _get_faiss()
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized)
        self.index.add(embeddings)

        # Persist
        faiss.write_index(self.index, str(self.INDEX_PATH))
        with open(self.CHUNKS_PATH, "w") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k most relevant document chunks."""
        q_embed = self._embed([query]).astype("float32")
        scores, indices = self.index.search(q_embed, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "score": float(score),
                "source": chunk.metadata.get("source", "Unknown"),
                "title": chunk.metadata.get("title", ""),
                "authors": chunk.metadata.get("authors", ""),
                "year": chunk.metadata.get("year", ""),
                "domain": chunk.metadata.get("domain", ""),
            })
        return results

    # ── Generation ────────────────────────────────────────────────────────────

    def _load_llm(self):
        if self._llm is None:
            from transformers import pipeline as hf_pipeline
            self._llm = hf_pipeline(
                "text2text-generation",
                model=self.llm_model_name,
                max_new_tokens=512,
                temperature=0.3,
            )

    def _build_prompt(self, query: str, contexts: list[dict]) -> str:
        context_text = "\n\n".join([
            f"[Source {i+1}: {c['title']} ({c['year']})]\n{c['text']}"
            for i, c in enumerate(contexts)
        ])
        return f"""You are an expert AI assistant specializing in railroad condition monitoring,
distributed acoustic sensing (DAS), and AI-driven manufacturing systems.

Answer the question based on the provided research context. Be precise and technical.
If the answer is not in the context, say so clearly.

Context:
{context_text}

Question: {query}

Answer:"""

    def query(
        self,
        question: str,
        top_k: int = 3,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Full RAG pipeline: retrieve → prompt → generate."""
        sources = self.retrieve(question, top_k=top_k)

        if not sources:
            return {
                "answer": "No relevant documents found. Please add documents to the knowledge base.",
                "sources": [],
            }

        prompt = self._build_prompt(question, sources)

        try:
            self._load_llm()
            result = self._llm(prompt, temperature=temperature)
            answer = result[0]["generated_text"].strip()
        except Exception as e:
            # Graceful fallback — return context directly
            answer = self._fallback_answer(question, sources)

        return {"answer": answer, "sources": sources}

    def _fallback_answer(self, question: str, sources: list[dict]) -> str:
        """Context-only fallback when LLM is unavailable."""
        top = sources[0]
        return (
            f"**Based on retrieved context from '{top['title']}' ({top['year']}):**\n\n"
            f"{top['text']}\n\n"
            f"*Retrieved {len(sources)} relevant passages. "
            f"For full LLM-generated answers, ensure the model is loaded.*"
        )
