"""Cost-aware routing: decide whether a query needs the large model.

The router is a transparent feature-based scorer rather than a learned
classifier, on purpose: it costs ~0.05 ms, needs no training data, and every
decision comes with human-readable reasons that show up on the dashboard.
`bench/run_bench.py` measures its accuracy against a labelled set, which is how
you would justify replacing it with a trained classifier later (same interface:
`score(query) -> RouteDecision`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_REASONING = re.compile(
    r"\b(why|explain|compare|contrast|analy[sz]e|trade-?offs?|design|architect\w*|prove|derive|"
    r"step[- ]by[- ]step|debug|optimi[sz]e|evaluate|critique|pros and cons|implications?|"
    r"strategy|recommend|reason about|walk me through|refactor)\b",
    re.I,
)
_CODE = re.compile(r"```|\bdef |\bclass |Traceback|\berror:|\bException\b|=>|\{\s*\}|;\s*$", re.M)
_MATH = re.compile(r"\b(calculate|solve|integral|derivative|probability|equation|complexity of)\b|\d+\s*[-+*/^]\s*\d+", re.I)
_SIMPLE = re.compile(r"^\s*(what is|what's|who is|who was|when (is|was|did)|where is|define|meaning of|how many)\b", re.I)
_ENUM = re.compile(r"(^|\n)\s*(\d+[.)]|-|\*)\s+", re.M)


@dataclass
class RouteDecision:
    tier: str  # "small" | "large"
    score: float
    reasons: list[str] = field(default_factory=list)


class ComplexityRouter:
    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold

    def score(self, query: str, embedding=None) -> RouteDecision:  # embedding unused; uniform interface
        s, reasons = 0.0, []
        words = len(query.split())

        if words > 40:
            s += 0.25
            reasons.append(f"long ({words} words)")
        if words > 90:
            s += 0.15
        cues = sorted({m.group(0).lower() for m in _REASONING.finditer(query)})
        if cues:
            # One cue is already a strong signal; extra cues add a little, capped.
            s += min(0.6, 0.45 + 0.1 * (len(cues) - 1))
            reasons.append("reasoning cue: " + ", ".join(f"'{c}'" for c in cues))
        if _CODE.search(query):
            s += 0.3
            reasons.append("contains code")
        if _MATH.search(query):
            s += 0.2
            reasons.append("math")
        if query.count("?") > 1 or len(_ENUM.findall(query)) >= 2:
            s += 0.2
            reasons.append("multi-part")
        if _SIMPLE.search(query) and words <= 12:
            s -= 0.25
            reasons.append("simple factual form")
        elif words <= 6:
            s -= 0.1
            reasons.append("very short")

        s = max(0.0, min(1.0, s))
        tier = "large" if s >= self.threshold else "small"
        return RouteDecision(tier, round(s, 3), reasons or ["no complexity signals"])


class EmbeddingRouter:
    """Routes by similarity to labelled examples, reusing the cache's query embedding.

    Score = sigmoid(k * (mean top-3 similarity to 'hard' examples
                         - mean top-3 similarity to 'simple' examples)).
    Comparing per-class top-3 means keeps it balanced even though the seed set
    has 3x more simple than hard examples. Costs one small matmul.
    """

    def __init__(self, embedder, examples, threshold: float = 0.5, top: int = 3, sharpness: float = 25.0):
        self.embedder, self.threshold, self.top, self.sharpness = embedder, threshold, top, sharpness
        self._labels = np.array([t for t, _ in examples])
        self._matrix = np.vstack([embedder.embed(q) for _, q in examples])
        self._queries = [q for _, q in examples]

    def score(self, query: str, embedding: np.ndarray | None = None) -> RouteDecision:
        emb = self.embedder.embed(query) if embedding is None else embedding
        sims = self._matrix @ emb
        means = {}
        for label in ("hard", "simple"):
            s = np.sort(sims[self._labels == label])[::-1][: self.top]
            means[label] = float(s.mean())
        p = 1 / (1 + np.exp(-self.sharpness * (means["hard"] - means["simple"])))
        nearest = self._queries[int(np.argmax(sims))]
        tier = "large" if p >= self.threshold else "small"
        return RouteDecision(tier, round(float(p), 3), [f"knn p(hard)={p:.2f}, nearest: {nearest[:60]!r}"])


class HybridRouter:
    """Large if *either* the heuristic or the kNN router says large.

    The heuristic is precise on explicit cues ("explain", code, math); the kNN
    router covers hard questions phrased without them. Taking the max trades
    some extra large-model calls for fewer hard questions sent to the small model.
    """

    def __init__(self, heuristic: ComplexityRouter, knn: EmbeddingRouter):
        self.heuristic, self.knn = heuristic, knn
        self.threshold = heuristic.threshold

    def score(self, query: str, embedding: np.ndarray | None = None) -> RouteDecision:
        h = self.heuristic.score(query)
        k = self.knn.score(query, embedding)
        tier = "large" if "large" in (h.tier, k.tier) else "small"
        return RouteDecision(tier, max(h.score, k.score), h.reasons + k.reasons)
