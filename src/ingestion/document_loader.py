"""
Document Loader — ingests PDF/TXT documents from the data/papers directory.
Chunks documents with overlap for better retrieval context.
"""

from __future__ import annotations
import re
from pathlib import Path


def _chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:  # skip trivially short chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _clean_text(text: str) -> str:
    """Basic text cleanup — remove excessive whitespace and encoding artefacts."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def load_pdf(path: Path) -> list[dict]:
    """Load a PDF and return chunked text with metadata."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        full_text = " ".join(page.get_text() for page in doc)
        full_text = _clean_text(full_text)
        chunks = _chunk_text(full_text)
        meta = {
            "source": path.name,
            "title": path.stem.replace("_", " ").replace("-", " "),
            "authors": "Rahman MA et al.",
            "year": "2024",
            "domain": "research_paper",
        }
        return [{"text": c, "metadata": meta} for c in chunks]
    except ImportError:
        print("PyMuPDF not installed — skipping PDF loading. Run: pip install pymupdf")
        return []
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def load_txt(path: Path) -> list[dict]:
    """Load a plain text file and return chunked content."""
    try:
        text = _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        chunks = _chunk_text(text)
        meta = {
            "source": path.name,
            "title": path.stem.replace("_", " "),
            "authors": "",
            "year": "",
            "domain": "document",
        }
        return [{"text": c, "metadata": meta} for c in chunks]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def load_all_documents(docs_dir: Path) -> list:
    """Load all supported documents from a directory."""
    from src.retrieval.rag_pipeline import DocumentChunk

    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []

    all_chunks = []
    loaders = {".pdf": load_pdf, ".txt": load_txt}

    for ext, loader in loaders.items():
        for fpath in docs_dir.glob(f"*{ext}"):
            print(f"Loading: {fpath.name}")
            raw = loader(fpath)
            for item in raw:
                all_chunks.append(
                    DocumentChunk(text=item["text"], metadata=item["metadata"])
                )

    print(f"Loaded {len(all_chunks)} chunks from {docs_dir}")
    return all_chunks
