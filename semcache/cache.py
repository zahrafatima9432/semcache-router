"""Two-tier answer cache: exact match first, then embedding similarity.

Tier 1 (exact): hash of the normalised query. Costs microseconds and never
needs an embedding, so a repeated question skips the embedder entirely.

Tier 2 (semantic): cosine similarity against every stored query embedding.
The store is kept in SQLite for persistence and mirrored into an in-memory
numpy matrix for search -- a brute-force matmul over 100k x 384 floats is a few
milliseconds, so a vector DB is not needed at portfolio scale. Swapping in
pgvector/HNSW is a drop-in change to `_search`.

The entity guard exists because embedding similarity is blind to the one word
that matters: "capital of France" and "capital of Germany" score ~0.9 with most
sentence encoders. Serving France's answer to the Germany question is the worst
failure a semantic cache can have, so a semantic hit is rejected when the two
queries mention different numbers or capitalised entities.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize(query: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", query.lower())).strip()


def query_hash(namespace: str, query: str) -> str:
    return hashlib.sha256(f"{namespace}\x00{normalize(query)}".encode()).hexdigest()


_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"[\"'`]([^\"'`]+)[\"'`]")
_CAPWORD = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*\b")
# Capitalised words that are just sentence starters / question words, not entities.
_NOT_ENTITIES = {
    "what", "who", "when", "where", "why", "how", "which", "is", "are", "can", "could",
    "should", "would", "do", "does", "did", "the", "a", "an", "i", "in", "on", "explain",
    "tell", "give", "list", "describe", "define", "compare", "please", "write", "show",
}


def key_tokens(query: str) -> frozenset[str]:
    """Numbers, quoted strings and capitalised words -- the tokens that change the answer."""
    tokens = set(_NUMBER.findall(query))
    tokens.update(q.lower() for q in _QUOTED.findall(query))
    tokens.update(w.lower() for w in _CAPWORD.findall(query) if w.lower() not in _NOT_ENTITIES)
    return frozenset(tokens)


def entities_conflict(a: str, b: str) -> str | None:
    """Return the offending token if one query names something the other doesn't.

    Checked against the *normalised text* of the other query rather than its key
    tokens, so casing differences don't count: "capital of japan" vs
    "What is the capital of Japan?" passes, "...France?" vs "...Germany?" fails.
    """
    for src, other in ((a, b), (b, a)):
        hay = f" {normalize(other)} "
        for tok in key_tokens(src):
            if f" {normalize(tok)} " not in hay:
                return tok
    return None


@dataclass
class CacheEntry:
    id: int
    namespace: str
    query: str
    response: str
    model: str
    tokens_in: int
    tokens_out: int
    created_at: float


@dataclass
class CacheHit:
    kind: str  # "exact" | "semantic"
    entry: CacheEntry
    similarity: float


class SemanticCache:
    def __init__(
        self,
        db_path: str,
        embedder,
        similarity_threshold: float = 0.90,
        ttl_s: int = 7 * 24 * 3600,
        entity_guard: bool = True,
    ):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.ttl_s = ttl_s
        self.entity_guard = entity_guard
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                qhash TEXT NOT NULL,
                query TEXT NOT NULL,
                embedding BLOB NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS ix_cache_qhash ON cache_entries(qhash)")
        self._db.commit()
        self._load()

    # ---- in-memory index -------------------------------------------------
    def _load(self) -> None:
        rows = self._db.execute(
            "SELECT id, namespace, qhash, query, embedding, response, model, tokens_in, tokens_out, created_at "
            "FROM cache_entries ORDER BY id"
        ).fetchall()
        self._entries: list[CacheEntry] = []
        self._by_hash: dict[str, int] = {}  # qhash -> position in self._entries
        vectors = []
        for r in rows:
            self._by_hash[r[2]] = len(self._entries)
            self._entries.append(CacheEntry(r[0], r[1], r[3], r[5], r[6], r[7], r[8], r[9]))
            vectors.append(np.frombuffer(r[4], dtype=np.float32))
        dim = self.embedder.dim
        self._matrix = np.vstack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
        self._namespaces = np.array([e.namespace for e in self._entries], dtype=object)

    def _fresh(self, entry: CacheEntry, now: float) -> bool:
        return now - entry.created_at <= self.ttl_s

    # ---- public API ------------------------------------------------------
    def lookup_exact(self, query: str, namespace: str = "default") -> CacheHit | None:
        with self._lock:
            pos = self._by_hash.get(query_hash(namespace, query))
            if pos is None:
                return None
            entry = self._entries[pos]
            if not self._fresh(entry, time.time()):
                return None
            return CacheHit("exact", entry, 1.0)

    def lookup_semantic(
        self, query: str, embedding: np.ndarray, namespace: str = "default"
    ) -> tuple[CacheHit | None, float, str | None]:
        """Returns (hit, best_similarity, rejection_reason)."""
        with self._lock:
            if not self._entries:
                return None, 0.0, None
            sims = self._matrix @ embedding
            sims = np.where(self._namespaces == namespace, sims, -1.0)
            now = time.time()
            best = float(sims.max())
            rejection = None
            # Walk the top candidates best-first: an expired or guard-rejected
            # neighbour shouldn't hide a valid one just below it.
            for pos in np.argsort(-sims)[:5]:
                sim = float(sims[pos])
                if sim < self.threshold:
                    break
                entry = self._entries[pos]
                if not self._fresh(entry, now):
                    continue
                if self.entity_guard and (tok := entities_conflict(query, entry.query)):
                    rejection = rejection or f"guard: '{tok}' differs from cached {entry.query!r} (sim {sim:.3f})"
                    continue
                return CacheHit("semantic", entry, sim), sim, None
            return None, best, rejection

    def put(
        self,
        query: str,
        embedding: np.ndarray,
        response: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        namespace: str = "default",
    ) -> CacheEntry:
        qh = query_hash(namespace, query)
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO cache_entries (namespace, qhash, query, embedding, response, model, tokens_in, tokens_out, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (namespace, qh, query, embedding.astype(np.float32).tobytes(), response, model, tokens_in, tokens_out, now),
            )
            self._db.commit()
            entry = CacheEntry(cur.lastrowid, namespace, query, response, model, tokens_in, tokens_out, now)
            self._by_hash[qh] = len(self._entries)
            self._entries.append(entry)
            self._matrix = np.vstack([self._matrix, embedding.astype(np.float32)[None, :]])
            self._namespaces = np.append(self._namespaces, namespace)
            return entry

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM cache_entries")
            self._db.commit()
        self._load()

    def close(self) -> None:
        self._db.close()

    def __len__(self) -> int:
        return len(self._entries)
