# -*- coding: utf-8 -*-
"""
Run:  python rag/index_docs.py
Indexes files from ../../data/company_docs into FAISS + meta.json
"""
import os, json, glob
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT.parent / "data" / "company_docs"
STORE_DIR = ROOT / "rag" / "vector_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80

def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        r = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in r.pages)
    if path.suffix.lower() in [".docx", ".doc"]:
        d = Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def main():
    files = []
    for pat in ("*.txt", "*.md", "*.pdf", "*.docx"):
        files += glob.glob(str(DOCS_DIR / pat))

    if not files:
        print(f"No files in {DOCS_DIR}")
        return

    model = SentenceTransformer(EMBED_MODEL)
    all_chunks: List[str] = []
    meta: List[Dict] = []

    for f in files:
        p = Path(f)
        text = read_text(p)
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            meta.append({"source": p.name, "chunk_id": idx})
            all_chunks.append(ch)

    print(f"Total chunks: {len(all_chunks)}")
    embs = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    with open(STORE_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "chunks": all_chunks}, f, ensure_ascii=False)

    print("Index saved.")

if __name__ == "__main__":
    main()
