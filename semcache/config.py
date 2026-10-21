"""Model price table and runtime settings.

Prices are USD per 1M tokens, taken from Groq's published on-demand rates
(Oct 2026). Edit them here if you switch providers -- every cost number in the
system is derived from this table, nothing else is hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_per_m: float
    output_per_m: float

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_m + tokens_out * self.output_per_m) / 1_000_000


SMALL_MODEL = ModelSpec("openai/gpt-oss-20b", input_per_m=0.075, output_per_m=0.30)
LARGE_MODEL = ModelSpec("openai/gpt-oss-120b", input_per_m=0.15, output_per_m=0.60)


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: os.getenv("SEMCACHE_DB", "semcache.db"))
    # Cosine similarity at or above which a stored answer is reused.
    similarity_threshold: float = field(default_factory=lambda: _env_float("SEMCACHE_THRESHOLD", 0.90))
    # Reject semantic hits whose numbers / named entities differ (see cache.key_tokens).
    entity_guard: bool = field(default_factory=lambda: os.getenv("SEMCACHE_GUARD", "1") == "1")
    cache_ttl_s: int = field(default_factory=lambda: int(os.getenv("SEMCACHE_TTL", 7 * 24 * 3600)))
    # Complexity score at or above which a query goes to the large model.
    route_threshold: float = field(default_factory=lambda: _env_float("SEMCACHE_ROUTE_THRESHOLD", 0.45))
    router: str = field(default_factory=lambda: os.getenv("SEMCACHE_ROUTER", "hybrid"))  # heuristic | hybrid
    embedder: str =field(default_factory=lambda: os.getenv("SEMCACHE_EMBEDDER", "bge"))  # bge | hash
    llm_backend: str = field(default_factory=lambda: os.getenv("SEMCACHE_LLM", "mock"))  # mock | groq
    small: ModelSpec = SMALL_MODEL
    large: ModelSpec = LARGE_MODEL
