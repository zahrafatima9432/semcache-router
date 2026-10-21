"""LLM backends. Both return token usage so cost can be computed from the price table.

`MockLLM` makes the whole system runnable (tests, benchmark, dashboard demo)
without an API key. Its latency is *simulated* from rough per-model figures and
reported, not slept, unless `sleep=True`. Benchmark numbers produced with it
measure cache/routing behaviour and cost accounting, not real model latency.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from .config import ModelSpec

SYSTEM_PROMPT = "You are a concise, accurate assistant. Answer directly."


@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


def _approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))


class MockLLM:
    # (base ms, ms per output token) -- rough shape of a small vs large hosted model.
    _LATENCY = {"small": (180.0, 2.0), "large": (450.0, 5.0)}

    def __init__(self, small: ModelSpec, large: ModelSpec, sleep: bool = False):
        self._tier_of = {small.name: "small", large.name: "large"}
        self.sleep = sleep

    def complete(self, model: ModelSpec, query: str) -> LLMResult:
        tier = self._tier_of.get(model.name, "large")
        tokens_in = _approx_tokens(SYSTEM_PROMPT) + _approx_tokens(query)
        tokens_out = 60 if tier == "small" else 180  # large models tend to answer at length
        base, per_tok = self._LATENCY[tier]
        latency = base + per_tok * tokens_out
        if self.sleep:
            time.sleep(latency / 1000)
        text = f"[mock {model.name}] answer to: {query}"
        return LLMResult(text, tokens_in, tokens_out, latency)


class GroqLLM:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (or use SEMCACHE_LLM=mock)")
        self._client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    def complete(self, model: ModelSpec, query: str) -> LLMResult:
        t0 = time.perf_counter()
        resp = self._client.post(
            self.URL,
            json={
                "model": model.name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            },
        )
        latency = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"],
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=latency,
        )


def make_llm(backend: str, small: ModelSpec, large: ModelSpec):
    if backend == "mock":
        return MockLLM(small, large)
    if backend == "groq":
        return GroqLLM()
    raise ValueError(f"unknown LLM backend {backend!r} (expected 'mock' or 'groq')")
