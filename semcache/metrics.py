"""Per-request cost/latency log and the aggregates the dashboard shows.

Every request records two costs:
  cost_usd          -- what this request actually cost (0 for cache hits)
  baseline_cost_usd -- what it would have cost with no cache and no router,
                       i.e. always calling the large model.
Savings = 1 - sum(cost) / sum(baseline). For a cache hit the baseline uses the
token counts of the original call, priced at the large model.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RequestRecord:
    query: str
    namespace: str
    path: str  # exact_hit | semantic_hit | small | large
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    lookup_ms: float  # time spent in cache lookups + embedding + routing
    similarity: float | None = None
    route_score: float | None = None
    note: str | None = None
    ts: float = 0.0


PATHS = ("exact_hit", "semantic_hit", "small", "large")


class RequestLog:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, query TEXT, namespace TEXT, path TEXT, model TEXT,
                tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, baseline_cost_usd REAL,
                latency_ms REAL, lookup_ms REAL, similarity REAL, route_score REAL, note TEXT
            )"""
        )
        self._db.commit()

    def record(self, rec: RequestRecord) -> RequestRecord:
        rec.ts = rec.ts or time.time()
        d = asdict(rec)
        cols = ", ".join(d)
        with self._lock:
            self._db.execute(f"INSERT INTO requests ({cols}) VALUES ({', '.join('?' * len(d))})", tuple(d.values()))
            self._db.commit()
        return rec

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            cur = self._db.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))
            names = [c[0] for c in cur.description]
            return [dict(zip(names, row)) for row in cur.fetchall()]

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM requests")
            self._db.commit()

    def close(self) -> None:
        self._db.close()

    def summary(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, path, cost_usd, baseline_cost_usd, latency_ms, lookup_ms FROM requests ORDER BY id"
            ).fetchall()
        n = len(rows)
        if n == 0:
            return {"requests": 0, "paths": {p: 0 for p in PATHS}, "latency": {}, "series": []}

        ts, path, cost, base, lat, lookup = (np.array(c) for c in zip(*rows))
        cost, base, lat, lookup = (a.astype(float) for a in (cost, base, lat, lookup))
        total_cost, total_base = float(cost.sum()), float(base.sum())

        latency = {}
        for p in PATHS:
            m = path == p
            if m.any():
                latency[p] = {
                    "count": int(m.sum()),
                    "p50_ms": round(float(np.percentile(lat[m], 50)), 2),
                    "p95_ms": round(float(np.percentile(lat[m], 95)), 2),
                    "avg_cost_usd": float(cost[m].mean()),
                }

        # Cumulative series for the cost chart, downsampled to <= 200 points.
        idx = np.unique(np.linspace(0, n - 1, min(n, 200)).astype(int))
        cum_cost, cum_base = np.cumsum(cost), np.cumsum(base)
        hits_cum = np.cumsum(np.isin(path, ["exact_hit", "semantic_hit"]))
        series = [
            {
                "i": int(i + 1),
                "ts": float(ts[i]),
                "cost": float(cum_cost[i]),
                "baseline": float(cum_base[i]),
                "hit_rate": float(hits_cum[i] / (i + 1)),
            }
            for i in idx
        ]

        hits = int(np.isin(path, ["exact_hit", "semantic_hit"]).sum())
        return {
            "requests": n,
            "paths": {p: int((path == p).sum()) for p in PATHS},
            "hit_rate": hits / n,
            "total_cost_usd": total_cost,
            "baseline_cost_usd": total_base,
            "savings_pct": (1 - total_cost / total_base) * 100 if total_base else 0.0,
            "cost_per_1k_requests_usd": total_cost / n * 1000,
            "p50_latency_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
            "avg_lookup_ms": round(float(lookup.mean()), 3),
            "latency": latency,
            "series": series,
        }
