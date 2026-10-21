"""Benchmark: threshold sweep for the cache, accuracy for the router, cost vs baseline.

    python -m bench.run_bench                     # bge-small embeddings, mock LLM
    python -m bench.run_bench --embedder hash     # offline fallback (not a semantic result!)

Writes bench/results.md and prints the same tables.

Metrics
  true-hit rate   : of queries whose intent was already answered, the share served
                    from cache with the *right* intent (higher = more savings)
  false hits      : cache hits that returned a different intent's answer
                    (each one is a wrong answer shown to a user -- want 0)
  false-hit rate  : false hits / all cache hits
  savings         : 1 - spend / always-large-no-cache spend, from the price table
The LLM is mocked, so latency numbers are simulated and only the cache-lookup
overhead is real. Hit rates, false hits, routing accuracy and cost accounting do
not depend on the LLM and are real measurements of this code on this workload.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.workload import ROUTING_HELDOUT, build_workload, routing_labels  # noqa: E402
from semcache import Gateway, Settings  # noqa: E402
from semcache.embedder import make_embedder  # noqa: E402
from semcache.route_seeds import ROUTE_SEEDS  # noqa: E402
from semcache.router import ComplexityRouter, EmbeddingRouter, HybridRouter  # noqa: E402


def run_cache(embedder, threshold: float, guard: bool, workload) -> dict:
    with tempfile.TemporaryDirectory() as d:
        s = Settings(db_path=os.path.join(d, "b.db"), similarity_threshold=threshold, entity_guard=guard, llm_backend="mock")
        gw = Gateway.from_settings(s, embedder=embedder)
        query_intent = {}
        seen_intents = set()
        eligible = true_hits = false_hits = exact = semantic = 0
        examples = []
        for q, intent, _tier in workload:
            ans = gw.ask(q)
            if intent in seen_intents:
                eligible += 1
            if ans.path in ("exact_hit", "semantic_hit"):
                exact += ans.path == "exact_hit"
                semantic += ans.path == "semantic_hit"
                if query_intent.get(ans.matched_query) == intent:
                    true_hits += 1
                else:
                    false_hits += 1
                    if len(examples) < 4:
                        examples.append(f"{q!r} got the answer for {ans.matched_query!r} (sim {ans.similarity})")
            else:
                query_intent[q] = intent
            seen_intents.add(intent)
        summ = gw.log.summary()
        gw.close()
        hits = exact + semantic
        return {
            "threshold": threshold, "guard": guard,
            "true_hit_rate": true_hits / eligible if eligible else 0,
            "false_hits": false_hits, "false_hit_rate": false_hits / hits if hits else 0,
            "exact": exact, "semantic": semantic, "savings_pct": summ["savings_pct"],
            "avg_lookup_ms": summ["avg_lookup_ms"], "examples": examples,
        }


def run_router(router, labels, embedder=None) -> dict:
    tp = fp = fn = tn = 0
    wrong = []
    for q, tier in labels:
        pred = router.score(q, embedder.embed(q) if embedder else None).tier
        if pred == "large" and tier == "hard": tp += 1
        elif pred == "large": fp += 1
        elif tier == "hard": fn += 1; wrong.append(q)
        else: tn += 1
    return {"accuracy": (tp + tn) / len(labels),
            "hard_recall": tp / (tp + fn) if tp + fn else 0,
            "large_share": (tp + fp) / len(labels), "fp": fp, "fn": fn, "missed": wrong[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge", choices=["bge", "hash"])
    ap.add_argument("--thresholds", default="0.80,0.85,0.88,0.90,0.92,0.95")
    args = ap.parse_args()

    t0 = time.time()
    embedder = make_embedder(args.embedder)
    workload = build_workload()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    lines = [f"# Benchmark results\n",
             f"Embedder: `{type(embedder).__name__}` · LLM: mock · {len(workload)} requests "
             f"({len(set(i for _, i, _ in workload))} distinct intents)\n"]
    if args.embedder == "hash":
        lines.append("> **Hashing embedder — surface-overlap only. These are not semantic-cache numbers.**\n")

    lines += ["## Cache: similarity threshold sweep\n",
              "| threshold | guard | true-hit rate | false hits | false-hit rate | exact / semantic hits | savings | avg lookup |",
              "|---|---|---|---|---|---|---|---|"]
    all_examples = []
    for th in thresholds:
        for guard in (False, True):
            r = run_cache(embedder, th, guard, workload)
            lines.append(f"| {th:.2f} | {'on' if guard else 'off'} | {r['true_hit_rate']:.1%} | {r['false_hits']} | "
                         f"{r['false_hit_rate']:.1%} | {r['exact']} / {r['semantic']} | {r['savings_pct']:.1f}% | {r['avg_lookup_ms']:.2f} ms |")
            if r["examples"]:
                all_examples.append((th, guard, r["examples"]))

    lines += ["\n### Sample false hits\n"]
    for th, guard, ex in all_examples[:4]:
        lines.append(f"- threshold {th:.2f}, guard {'on' if guard else 'off'}:")
        lines += [f"  - {e}" for e in ex[:2]]

    lines += ["\n## Router: accuracy against hand labels\n",
              "- **in-sample**: the workload queries. The heuristic rules were written alongside them and the kNN "
              "router uses them as its labelled examples, so only the heuristic row is shown and it is optimistic.",
              "- **held-out**: 24 queries written afterwards; hard ones avoid trigger words, some easy ones contain them. "
              "This is the honest number.\n",
              "| router | set | accuracy | hard-query recall | share sent to large | false large | missed hard |",
              "|---|---|---|---|---|---|---|"]
    heldout = [(q, t) for t, q in ROUTING_HELDOUT]
    heur = ComplexityRouter(0.45)
    knn = EmbeddingRouter(embedder, ROUTE_SEEDS)
    hybrid = HybridRouter(heur, knn)
    runs = [("heuristic", "in-sample", heur, routing_labels(), None),
            ("heuristic", "held-out", heur, heldout, None),
            ("kNN", "held-out", knn, heldout, embedder),
            ("hybrid", "held-out", hybrid, heldout, embedder)]
    misses = {}
    for name, split, router, labels, emb in runs:
        r = run_router(router, labels, emb)
        misses[(name, split)] = r["missed"]
        lines.append(f"| {name} | {split} | {r['accuracy']:.1%} | {r['hard_recall']:.1%} | {r['large_share']:.1%} | {r['fp']} | {r['fn']} |")
    for key in (("heuristic", "held-out"), ("hybrid", "held-out")):
        if misses[key]:
            lines.append(f"\nHard held-out queries the {key[0]} router sent to the small model (first 3):")
            lines += [f"- {q}" for q in misses[key]]

    lines.append(f"\n_Run took {time.time() - t0:.1f}s._\n")
    out = "\n".join(lines)
    print(out)
    Path(__file__).with_name("results.md" if args.embedder == "bge" else "results_hash.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
