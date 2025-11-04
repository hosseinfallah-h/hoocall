# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / "rag" / "vector_store"

EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(STORE_DIR / "faiss.index"))
        with open(STORE_DIR / "meta.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.meta = data["meta"]
        self.chunks = data["chunks"]

    def search(self, query: str, k: int = TOP_K) -> List[Tuple[str, dict, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1: 
                continue
            out.append((self.chunks[idx], self.meta[idx], float(score)))
        return out
