"""The request pipeline: exact cache -> semantic cache -> router -> model -> store.

    query
      |-- exact hash hit? ------------------> return cached   (no embedding, ~0.01 ms)
      |-- embed (once, reused for insert)
      |-- similarity >= threshold
      |     and entity guard passes? -------> return cached   (~5-15 ms with bge on CPU)
      |-- router.score(query, embedding)    (heuristic cues + kNN over labelled examples)
      |-- call small or large model
      |-- cache.put(query, embedding, answer)
      `-- log cost, baseline cost, latency
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .cache import SemanticCache
from .config import ModelSpec, Settings
from .embedder import make_embedder
from .llm import make_llm
from .metrics import RequestLog, RequestRecord
from .router import ComplexityRouter, EmbeddingRouter, HybridRouter


@dataclass
class Answer:
    answer: str
    path: str
    model: str
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    similarity: float | None = None
    matched_query: str | None = None
    route_score: float | None = None
    route_reasons: list[str] | None = None
    note: str | None = None


class Gateway:
    def __init__(self, settings: Settings, embedder, cache: SemanticCache, router: ComplexityRouter, llm, log: RequestLog):
        self.settings = settings
        self.embedder = embedder
        self.cache = cache
        self.router = router
        self.llm = llm
        self.log = log

    @classmethod
    def from_settings(cls, settings: Settings | None = None, embedder=None, llm=None) -> "Gateway":
        s = settings or Settings()
        emb = embedder or make_embedder(s.embedder)
        cache = SemanticCache(s.db_path, emb, s.similarity_threshold, s.cache_ttl_s, s.entity_guard)
        router = ComplexityRouter(s.route_threshold)
        if s.router == "hybrid":
            from .route_seeds import ROUTE_SEEDS

            router = HybridRouter(router, EmbeddingRouter(emb, ROUTE_SEEDS))
        elif s.router != "heuristic":
            raise ValueError(f"unknown router {s.router!r} (expected 'heuristic' or 'hybrid')")
        return cls(s, emb, cache, router, llm or make_llm(s.llm_backend, s.small, s.large), RequestLog(s.db_path))

    def close(self) -> None:
        """Release the SQLite files (required on Windows before deleting them)."""
        self.cache.close()
        self.log.close()

    def _model_for(self, tier: str) -> ModelSpec:
        return self.settings.large if tier == "large" else self.settings.small

    def _cached(self, hit, query: str, namespace: str, t0: float, note: str | None = None) -> Answer:
        lookup_ms = (time.perf_counter() - t0) * 1000
        e = hit.entry
        baseline = self.settings.large.cost(e.tokens_in, e.tokens_out)
        path = f"{hit.kind}_hit"
        self.log.record(RequestRecord(
            query, namespace, path, e.model, 0, 0, 0.0, baseline, lookup_ms, lookup_ms,
            similarity=hit.similarity, note=note,
        ))
        return Answer(e.response, path, e.model, 0.0, baseline, lookup_ms,
                      similarity=round(hit.similarity, 4), matched_query=e.query, note=note)

    def ask(self, query: str, namespace: str = "default", force_tier: str | None = None) -> Answer:
        query = query.strip()
        if not query:
            raise ValueError("empty query")
        t0 = time.perf_counter()

        if hit := self.cache.lookup_exact(query, namespace):
            return self._cached(hit, query, namespace, t0)

        embedding = self.embedder.embed(query)
        hit, best_sim, rejection = self.cache.lookup_semantic(query, embedding, namespace)
        if hit:
            return self._cached(hit, query, namespace, t0)

        decision = self.router.score(query, embedding)
        tier = force_tier or decision.tier
        model = self._model_for(tier)
        lookup_ms = (time.perf_counter() - t0) * 1000

        result = self.llm.complete(model, query)
        cost = model.cost(result.tokens_in, result.tokens_out)
        # Baseline = these same tokens priced at the large model. When the small model
        # answered, its reply is usually shorter than the large model's would have been,
        # so this understates the baseline -- reported savings are a lower bound.
        baseline = self.settings.large.cost(result.tokens_in, result.tokens_out)
        self.cache.put(query, embedding, result.text, model.name, result.tokens_in, result.tokens_out, namespace)
        latency = lookup_ms + result.latency_ms
        self.log.record(RequestRecord(
            query, namespace, tier, model.name, result.tokens_in, result.tokens_out, cost, baseline,
            latency, lookup_ms, similarity=best_sim or None, route_score=decision.score, note=rejection,
        ))
        return Answer(result.text, tier, model.name, cost, baseline, latency,
                      similarity=round(best_sim, 4) if best_sim else None,
                      route_score=decision.score, route_reasons=decision.reasons, note=rejection)
