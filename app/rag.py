import json
import re
import threading
from pathlib import Path

import numpy as np

from .config import CONFIG, INDEX_DIR


_embed_model = None
_embed_lock = threading.Lock()


def get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    with _embed_lock:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(CONFIG["embedding_model"])
    return _embed_model


def unload_embedder() -> None:
    global _embed_model
    with _embed_lock:
        _embed_model = None
        import gc
        gc.collect()


def chunk_text(text: str, chunk_size: int | None = None,
               chunk_overlap: int | None = None) -> list[str]:
    chunk_size = int(chunk_size or CONFIG.get("chunk_size", 900))
    chunk_overlap = int(chunk_overlap or CONFIG.get("chunk_overlap", 120))
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def extract_text(path: Path, filename: str, mime: str = "", extra_text: str = "") -> str:
    path = Path(path)
    ext = Path(filename or path.name).suffix.lower()
    mime = (mime or "").lower()
    parts = []

    try:
        if ext in {".txt", ".md", ".markdown", ".py", ".bat", ".ps1", ".json", ".csv",
                   ".log", ".yaml", ".yml", ".ini", ".cfg", ".js", ".ts", ".html", ".htm"}:
            raw = path.read_text(encoding="utf-8", errors="ignore")[:500_000]
            if ext in {".html", ".htm"} or "html" in mime:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                raw = soup.get_text("\n")
            parts.append(raw)

        elif ext == ".pdf" or "pdf" in mime:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            parts.append("\n".join((p.extract_text() or "") for p in reader.pages)[:500_000])

        elif ext == ".docx" or "wordprocessingml" in mime:
            from docx import Document
            doc = Document(str(path))
            parts.append("\n".join(p.text for p in doc.paragraphs)[:500_000])

        elif mime.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp",
                                                  ".webp", ".tif", ".tiff"}:
            from PIL import Image
            image = Image.open(path)
            parts.append(f"Image attachment: {filename}; size={image.size}; mode={image.mode}")

    except Exception as exc:
        parts.append(f"Text extraction failed for {filename}: {exc}")

    if extra_text:
        parts.append(extra_text)

    return "\n".join(p for p in parts if p)[:500_000]


def _index_paths(project_id: int):
    return INDEX_DIR / f"{project_id}.faiss", INDEX_DIR / f"{project_id}.chunks.json"


def _load_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        return []
    try:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_chunks(chunks_path: Path, chunks: list[dict]) -> None:
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def add_texts(project_id: int, texts: list[str], metadatas: list[dict] | None = None) -> int:
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return 0

    import faiss

    model = get_embed_model()
    dim = model.get_sentence_embedding_dimension()
    index_path, chunks_path = _index_paths(project_id)

    chunks = _load_chunks(chunks_path)
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            if index.d != dim or index.ntotal != len(chunks):
                raise ValueError("index/chunk mismatch")
        except Exception:
            index, chunks = faiss.IndexFlatIP(dim), []
    else:
        index = faiss.IndexFlatIP(dim)

    embeddings = np.asarray(model.encode(texts, normalize_embeddings=True), dtype="float32")
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    metadatas = metadatas or [{} for _ in texts]
    for text, meta in zip(texts, metadatas):
        chunks.append({"text": text, "metadata": meta or {}})
    _save_chunks(chunks_path, chunks)
    return len(texts)


def rebuild_project(project_id: int) -> int:
    """Re-embed every stored file text for a project (after model switch)."""
    index_path, chunks_path = _index_paths(project_id)
    index_path.unlink(missing_ok=True)
    chunks_path.unlink(missing_ok=True)

    total = 0
    for row in db_files(project_id):
        text = row["text_extracted"] or ""
        chunks = chunk_text(text)
        if chunks:
            total += add_texts(project_id, chunks,
                               [{"file_id": row["id"], "filename": row["filename"],
                                 "mime": row["mime"]} for _ in chunks])
    return total


def db_files(project_id: int):
    from . import db
    return db.list_files(project_id)


def search_project(project_id: int, query: str, top_k: int | None = None) -> list[dict]:
    top_k = int(top_k or CONFIG.get("top_k_rag", 4))
    query = (query or "").strip()
    if not query:
        return []

    import faiss

    index_path, chunks_path = _index_paths(project_id)
    if not index_path.exists() or not chunks_path.exists():
        return []

    chunks = _load_chunks(chunks_path)
    if not chunks:
        return []

    try:
        index = faiss.read_index(str(index_path))
    except Exception:
        return []

    if index.ntotal == 0 or index.ntotal != len(chunks):
        return []

    model = get_embed_model()
    qv = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
    scores, ids = index.search(qv, min(top_k, index.ntotal))

    out = []
    for score, idx in zip(scores[0], ids[0]):
        idx = int(idx)
        if 0 <= idx < len(chunks):
            item = chunks[idx]
            out.append({"text": item.get("text", ""), "score": float(score),
                        "metadata": item.get("metadata", {})})
    return out
    
def remove_file(project_id: int, file_id: int) -> int:
    index_path, chunks_path = _index_paths(project_id)

    if not chunks_path.exists():
        return 0

    chunks = _load_chunks(chunks_path)
    remaining = []

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        if meta.get("file_id") != file_id:
            remaining.append(chunk)

    removed = len(chunks) - len(remaining)
    if removed <= 0:
        return 0

    if not remaining:
        index_path.unlink(missing_ok=True)
        chunks_path.unlink(missing_ok=True)
        return removed

    import faiss

    model = get_embed_model()
    dim = model.get_sentence_embedding_dimension()

    index = faiss.IndexFlatIP(dim)
    texts = [c.get("text", "") for c in remaining if c.get("text")]

    if texts:
        embeddings = np.asarray(
            model.encode(texts, normalize_embeddings=True),
            dtype="float32",
        )
        index.add(embeddings)

    faiss.write_index(index, str(index_path))
    _save_chunks(chunks_path, remaining)

    return removed