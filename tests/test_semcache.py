import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from semcache import Gateway, Settings
from semcache.api import create_app
from semcache.cache import SemanticCache, entities_conflict, normalize
from semcache.config import LARGE_MODEL, SMALL_MODEL
from semcache.embedder import HashingEmbedder
from semcache.router import ComplexityRouter


@pytest.fixture
def gw(tmp_path):
    s = Settings(db_path=str(tmp_path / "t.db"), similarity_threshold=0.80, llm_backend="mock")
    return Gateway.from_settings(s, embedder=HashingEmbedder())


# ---- cache ---------------------------------------------------------------
def test_normalize():
    assert normalize("  What IS the   capital, of Japan?? ") == "what is the capital of japan"


def test_exact_hit_ignores_case_and_punctuation(gw):
    first = gw.ask("What is the capital of Japan?")
    again = gw.ask("what is the capital of japan")
    assert first.path in ("small", "large")
    assert again.path == "exact_hit" and again.cost_usd == 0
    assert again.answer == first.answer


def test_semantic_hit_on_close_paraphrase(gw):
    gw.ask("How do I reverse a list in Python?")
    a = gw.ask("How do I reverse a list in Python please")
    assert a.path == "semantic_hit"
    assert a.matched_query == "How do I reverse a list in Python?"


def test_entity_guard_blocks_wrong_answer(tmp_path):
    s = Settings(db_path=str(tmp_path / "g.db"), similarity_threshold=0.5, llm_backend="mock", entity_guard=True)
    g = Gateway.from_settings(s, embedder=HashingEmbedder())
    g.ask("What is the capital of France?")
    a = g.ask("What is the capital of Germany?")
    assert a.path != "semantic_hit"
    assert a.note and "guard" in a.note


def test_without_guard_the_same_pair_is_a_false_hit(tmp_path):
    s = Settings(db_path=str(tmp_path / "ng.db"), similarity_threshold=0.5, llm_backend="mock", entity_guard=False)
    g = Gateway.from_settings(s, embedder=HashingEmbedder())
    g.ask("What is the capital of France?")
    assert g.ask("What is the capital of Germany?").path == "semantic_hit"  # documents the failure mode


def test_entities_conflict_rules():
    assert entities_conflict("capital of japan", "What is the capital of Japan?") is None
    assert entities_conflict("What is 15% of 200?", "What is 25% of 200?") is not None
    assert entities_conflict("Python 3.11 features", "Python 3.12 features") is not None


def test_namespace_isolation(gw):
    gw.ask("What is our refund policy?", namespace="tenant-a")
    assert gw.ask("What is our refund policy?", namespace="tenant-b").path in ("small", "large")
    assert gw.ask("What is our refund policy?", namespace="tenant-a").path == "exact_hit"


def test_ttl_expiry(tmp_path):
    emb = HashingEmbedder()
    c = SemanticCache(str(tmp_path / "ttl.db"), emb, similarity_threshold=0.8, ttl_s=1)
    q = "what is a docker container"
    c.put(q, emb.embed(q), "answer", "m", 10, 10)
    assert c.lookup_exact(q) is not None
    c._entries[0].created_at = time.time() - 5
    assert c.lookup_exact(q) is None
    assert c.lookup_semantic(q, emb.embed(q))[0] is None


def test_cache_persists_across_restarts(tmp_path):
    emb = HashingEmbedder()
    path = str(tmp_path / "p.db")
    SemanticCache(path, emb).put("hello world", emb.embed("hello world"), "hi", "m", 1, 1)
    c2 = SemanticCache(path, emb)
    assert len(c2) == 1 and c2.lookup_exact("Hello, world!") is not None
    assert np.allclose(c2._matrix[0], emb.embed("hello world"))


# ---- router --------------------------------------------------------------
@pytest.mark.parametrize("q", ["What is the capital of Japan?", "postgres default port", "Who wrote Hamlet?"])
def test_router_simple(q):
    assert ComplexityRouter(0.45).score(q).tier == "small"


@pytest.mark.parametrize("q", [
    "Explain the trade-offs between HNSW and IVF indexes for vector search.",
    "Design a rate limiter for a multi-tenant LLM API and explain your choices.",
    "Why does this fail?\n```python\ndef f(x): return x[0]\n```\nTraceback: IndexError",
])
def test_router_hard(q):
    d = ComplexityRouter(0.45).score(q)
    assert d.tier == "large" and d.reasons


# ---- cost accounting -----------------------------------------------------
def test_cost_and_savings_accounting(gw):
    a = gw.ask("What is the capital of Japan?")  # routed small
    assert a.path == "small" and a.model == SMALL_MODEL.name
    assert a.baseline_cost_usd > a.cost_usd > 0
    gw.ask("What is the capital of Japan?")  # exact hit, cost 0
    s = gw.log.summary()
    assert s["requests"] == 2 and s["paths"]["exact_hit"] == 1
    assert s["total_cost_usd"] == pytest.approx(a.cost_usd)
    assert s["baseline_cost_usd"] == pytest.approx(2 * a.baseline_cost_usd)
    assert 50 < s["savings_pct"] < 100


def test_force_tier(gw):
    a = gw.ask("What is 2+2?", force_tier="large")
    assert a.model == LARGE_MODEL.name and a.cost_usd == pytest.approx(a.baseline_cost_usd)


# ---- API -----------------------------------------------------------------
def test_api_roundtrip(gw):
    client = TestClient(create_app(gw))
    r = client.post("/ask", json={"query": "What is an embedding?"})
    assert r.status_code == 200 and r.json()["path"] in ("small", "large")
    assert client.post("/ask", json={"query": "what is an embedding"}).json()["path"] == "exact_hit"
    stats = client.get("/stats").json()
    assert stats["requests"] == 2 and stats["cache_entries"] == 1
    assert len(client.get("/requests").json()) == 2
    assert "Semantic cache" in client.get("/").text
    assert client.post("/ask", json={"query": ""}).status_code == 422
    client.post("/reset")
    assert client.get("/stats").json()["requests"] == 0


def test_hybrid_router_uses_embedding_examples():
    from semcache.router import EmbeddingRouter, HybridRouter

    emb = HashingEmbedder()
    seeds = [("hard", "Explain the trade-offs of distributed consensus protocols"),
             ("hard", "How should I shard a database with uneven tenant load"),
             ("simple", "What is the capital of Japan"), ("simple", "Who wrote Hamlet"),
             ("simple", "What does CPU stand for"), ("simple", "How many days in a year")]
    r = HybridRouter(ComplexityRouter(0.45), EmbeddingRouter(emb, seeds))
    # No trigger words, so the heuristic alone says small; the kNN vote says large.
    q = "How should I shard a database where some tenants have far more load"
    assert ComplexityRouter(0.45).score(q).tier == "small"
    assert r.score(q, emb.embed(q)).tier == "large"
    assert r.score("Who wrote Macbeth").tier == "small"
