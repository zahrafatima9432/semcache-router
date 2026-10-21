"""Replay the benchmark workload against a running server to populate the dashboard.

    uvicorn semcache.api:app            # terminal 1
    python -m bench.demo_traffic        # terminal 2, then open http://localhost:8000
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.workload import build_workload  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--limit", type=int, default=0, help="only send the first N queries (useful with a real LLM)")
args = ap.parse_args()

work = build_workload()
if args.limit:
    work = work[: args.limit]
with httpx.Client(base_url=args.url, timeout=120) as c:
    for i, (q, _, _) in enumerate(work, 1):
        a = c.post("/ask", json={"query": q}).json()
        print(f"{i:4d}  {a['path']:<13} {a['cost_usd']:.6f}  {q[:70]}")
print("done -- open the dashboard")
