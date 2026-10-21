"""Query embedders.

`BGEEmbedder` is the real one (bge-small-en-v1.5, 384-d, the same model the
RAG project uses). `HashingEmbedder` is a dependency-free fallback for tests and
offline runs: hashed word + character-trigram counts. It captures surface
overlap only, so it is much worse at paraphrases -- never benchmark with it and
report the numbers as semantic-cache results.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray:  # returns L2-normalised float32 vector
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        return int.from_bytes(hashlib.md5(feature.encode()).digest()[:4], "little") % self.dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = text.lower()
        for word in re.findall(r"\w+", text):
            vec[self._bucket("w:" + word)] += 1.0
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                vec[self._bucket("c:" + padded[i : i + 3])] += 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self._model = SentenceTransformer(model_name)
        # Renamed in newer sentence-transformers; support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, text: str) -> np.ndarray:
        # Query-to-query similarity, so no "Represent this sentence..." retrieval prefix.
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)


def make_embedder(kind: str) -> Embedder:
    if kind == "hash":
        return HashingEmbedder()
    if kind == "bge":
        return BGEEmbedder()
    raise ValueError(f"unknown embedder {kind!r} (expected 'bge' or 'hash')")
