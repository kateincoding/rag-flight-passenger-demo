"""Agent 2 — Retrieval: FAISS vector store + semantic top-k search.

Uses asymmetric embeddings (RETRIEVAL_DOCUMENT when indexing,
RETRIEVAL_QUERY when searching) and inner-product on L2-normalised vectors
(= cosine similarity).
"""
import json
from typing import Dict, List, Tuple

import faiss
import numpy as np

import config
from llm import embed


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class VectorStore:
    """Simple FAISS-based vector store for retrieval."""

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine on unit vectors
        self.chunks: List[Dict] = []

    def add(self, chunks: List[Dict]) -> None:
        vectors = []
        for chunk in chunks:
            emb = np.array(embed(chunk["text"], "RETRIEVAL_DOCUMENT"), dtype=np.float32)
            vectors.append(_normalize(emb))
            self.chunks.append(chunk)
        self.index.add(np.stack(vectors))

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Tuple[Dict, float]]:
        q = _normalize(np.array(embed(query, "RETRIEVAL_QUERY"), dtype=np.float32))
        scores, indices = self.index.search(q.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


def load_knowledge_base(path: str = config.KNOWLEDGE_PATH) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def build_vector_store() -> VectorStore:
    """Build and populate the vector store from the knowledge base on disk."""
    store = VectorStore()
    store.add(load_knowledge_base())
    return store


def retrieval_agent(store: VectorStore, query: str,
                    top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
    """Retrieve top-k most relevant chunks for the query."""
    return [
        {
            "chunk_id": chunk["id"],
            "category": chunk["category"],
            "text": chunk["text"],
            "score": round(score, 4),
        }
        for chunk, score in store.search(query, top_k=top_k)
    ]
