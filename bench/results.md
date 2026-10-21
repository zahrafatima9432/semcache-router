# Benchmark results

Embedder: `BGEEmbedder` · LLM: mock · 192 requests (48 distinct intents)

## Cache: similarity threshold sweep

| threshold | guard | true-hit rate | false hits | false-hit rate | exact / semantic hits | savings | avg lookup |
|---|---|---|---|---|---|---|---|
| 0.80 | off | 83.3% | 33 | 21.6% | 39 / 114 | 86.3% | 8.45 ms |
| 0.80 | on | 82.6% | 12 | 9.2% | 61 / 70 | 76.5% | 7.10 ms |
| 0.85 | off | 87.5% | 18 | 12.5% | 48 / 96 | 83.9% | 7.75 ms |
| 0.85 | on | 86.1% | 2 | 1.6% | 66 / 60 | 75.6% | 6.95 ms |
| 0.88 | off | 86.1% | 14 | 10.1% | 54 / 84 | 81.2% | 7.68 ms |
| 0.88 | on | 86.1% | 2 | 1.6% | 66 / 60 | 75.6% | 7.02 ms |
| 0.90 | off | 86.8% | 6 | 4.6% | 61 / 70 | 79.9% | 7.56 ms |
| 0.90 | on | 86.1% | 0 | 0.0% | 68 / 56 | 75.3% | 6.74 ms |
| 0.92 | off | 83.3% | 6 | 4.8% | 66 / 60 | 75.7% | 6.93 ms |
| 0.92 | on | 83.3% | 0 | 0.0% | 72 / 48 | 72.1% | 6.74 ms |
| 0.95 | off | 80.6% | 3 | 2.5% | 73 / 46 | 72.7% | 6.36 ms |
| 0.95 | on | 79.9% | 0 | 0.0% | 77 / 38 | 70.3% | 6.27 ms |

### Sample false hits

- threshold 0.80, guard off:
  - 'What is the time complexity of quicksort?' got the answer for 'What is the time complexity of binary search?' (sim 0.822)
  - 'Python 3.12 new features' got the answer for 'Python 3.11 new features' (sim 0.9706)
- threshold 0.80, guard on:
  - 'What is the time complexity of quicksort?' got the answer for 'What is the time complexity of binary search?' (sim 0.822)
  - 'How do I install pandas?' got the answer for 'How do I install numpy?' (sim 0.8189)
- threshold 0.85, guard off:
  - 'Python 3.12 new features' got the answer for 'Python 3.11 new features' (sim 0.9706)
  - 'How do I reverse a string in Python?' got the answer for 'How do I reverse a list in JavaScript?' (sim 0.8516)
- threshold 0.85, guard on:
  - 'How do I reverse a list in Python?' got the answer for 'How do I reverse a string in Python?' (sim 0.8912)
  - "What's the way to reverse a list in Python?" got the answer for 'How do I reverse a string in Python?' (sim 0.8881)

## Router: accuracy against hand labels

- **in-sample**: the workload queries. The heuristic rules were written alongside them and the kNN router uses them as its labelled examples, so only the heuristic row is shown and it is optimistic.
- **held-out**: 24 queries written afterwards; hard ones avoid trigger words, some easy ones contain them. This is the honest number.

| router | set | accuracy | hard-query recall | share sent to large | false large | missed hard |
|---|---|---|---|---|---|---|
| heuristic | in-sample | 100.0% | 100.0% | 25.0% | 0 | 0 |
| heuristic | held-out | 50.0% | 0.0% | 0.0% | 0 | 12 |
| kNN | held-out | 95.8% | 100.0% | 54.2% | 1 | 0 |
| hybrid | held-out | 95.8% | 100.0% | 54.2% | 1 | 0 |

Hard held-out queries the heuristic router sent to the small model (first 3):
- How would a CRDT keep a shared document consistent when two people edit the same line offline?
- What happens to gradient flow in a 50-layer plain network without residual connections?
- Is it safe to use read-committed isolation for an inventory reservation system?

_Run took 74.4s._
