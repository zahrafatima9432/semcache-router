"""Labelled benchmark workload.

PARAPHRASE_GROUPS: each group is one intent asked several ways. After the first
query in a group is answered, every later query in that group *should* be a
cache hit (a true hit). `tier` is the hand label for routing evaluation.

NEAR_MISS_GROUPS: queries that look alike but need different answers
("capital of France" vs "capital of Germany"). Each one is its own intent, so
any cache hit between them is a *false hit* -- a wrong answer served from cache.

The labels are mine and the set is small (hand-written, ~150 queries). Treat the
benchmark as a regression test for behaviour and a way to pick thresholds, not
as a claim about production traffic distributions.
"""

PARAPHRASE_GROUPS = [
    ("simple", ["What is the capital of Japan?", "capital of japan", "Which city is Japan's capital?", "Tell me the capital city of Japan"]),
    ("simple", ["How many days are in a leap year?", "number of days in a leap year", "How many days does a leap year have?"]),
    ("simple", ["What is HTTP status code 404?", "what does a 404 error mean", "Meaning of HTTP 404"]),
    ("simple", ["Who wrote Pride and Prejudice?", "Pride and Prejudice author", "Who is the author of Pride and Prejudice?"]),
    ("simple", ["What is the boiling point of water in Celsius?", "water boiling point celsius", "At what temperature does water boil in Celsius?"]),
    ("simple", ["What does RAG stand for in AI?", "RAG meaning in machine learning", "What is the full form of RAG in LLMs?"]),
    ("simple", ["How do I reverse a list in Python?", "python reverse a list", "What's the way to reverse a list in Python?", "reverse list python"]),
    ("simple", ["What is the default port for PostgreSQL?", "postgres default port", "Which port does PostgreSQL use by default?"]),
    ("simple", ["How do I undo the last git commit?", "undo last commit git", "How can I revert my most recent git commit?"]),
    ("simple", ["What is an embedding in NLP?", "define embedding in NLP", "What are text embeddings?"]),
    ("simple", ["How do I check my Python version?", "check python version", "command to see which Python version is installed"]),
    ("simple", ["What is the time complexity of binary search?", "binary search time complexity", "Big O of binary search"]),
    ("simple", ["How do I create a virtual environment in Python?", "create python venv", "How to make a Python virtual environment?"]),
    ("simple", ["What is the speed of light?", "speed of light in vacuum", "How fast does light travel?"]),
    ("simple", ["What is a Docker container?", "define docker container", "What are Docker containers?"]),
    ("hard", ["Explain the trade-offs between HNSW and IVF indexes for vector search.",
              "Compare HNSW vs IVF vector indexes and their trade-offs",
              "What are the pros and cons of HNSW compared to IVF for approximate nearest neighbour search?"]),
    ("hard", ["Why does a cross-encoder reranker improve retrieval quality over a bi-encoder, and what does it cost?",
              "Explain why cross-encoder reranking beats bi-encoder retrieval and the latency cost involved",
              "Analyze cross-encoder vs bi-encoder retrieval: why is reranking better and what is the trade-off?"]),
    ("hard", ["Design a rate limiter for a multi-tenant LLM API and explain your choices.",
              "How would you architect a rate limiter for a multi-tenant LLM API? Explain the design.",
              "Walk me through designing rate limiting for an LLM API with many tenants."]),
    ("hard", ["Explain step by step how backpropagation computes gradients in a neural network.",
              "Walk me through backpropagation step by step",
              "Explain how backprop calculates gradients, step-by-step."]),
    ("hard", ["Compare LoRA and full fine-tuning: when should you use each and why?",
              "LoRA vs full fine-tuning — explain when to use which and why",
              "What are the trade-offs between LoRA and full fine-tuning?"]),
    ("hard", ["Debug this: my FastAPI endpoint returns 422 when I send JSON with a nested list. Why?",
              "Why does my FastAPI endpoint give a 422 error when the JSON body has a nested list? Help me debug it.",
              "Help me debug a FastAPI 422 error on a request body containing a nested list"]),
    ("hard", ["Explain why semantic caching can return wrong answers and how to mitigate it.",
              "Why can a semantic cache serve incorrect answers, and how do you prevent that?",
              "Analyze the failure modes of semantic caching for LLMs and their mitigations"]),
    ("hard", ["Evaluate the implications of using eventual consistency for a payment ledger.",
              "What are the implications of eventual consistency in a payment ledger? Evaluate the risks.",
              "Critique using an eventually consistent database for payment ledgers"]),
]

NEAR_MISS_GROUPS = [
    ["What is the capital of France?", "What is the capital of Germany?", "What is the capital of Italy?"],
    ["What is HTTP status code 500?", "What is HTTP status code 503?"],
    ["How do I reverse a list in JavaScript?", "How do I reverse a string in Python?"],
    ["Who wrote Hamlet?", "Who wrote Macbeth?"],
    ["What is the default port for MySQL?", "What is the default port for Redis?"],
    ["How many days are in February 2024?", "How many days are in February 2023?"],
    ["What is 15% of 200?", "What is 25% of 200?"],
    ["Convert 100 USD to EUR", "Convert 100 USD to GBP"],
    ["What is the time complexity of quicksort?", "What is the time complexity of merge sort?"],
    ["How do I install numpy?", "How do I install pandas?"],
    ["What is the population of Pakistan?", "What is the population of India?"],
    ["Python 3.11 new features", "Python 3.12 new features"],
]


# Held-out routing set, written *after* the router rules and deliberately phrased
# so hard questions avoid the trigger words and some easy ones contain them.
# The in-sample set above was written alongside the rules, so its routing
# accuracy is optimistic by construction; this one is the honest number.
ROUTING_HELDOUT = [
    ("hard", "How would a CRDT keep a shared document consistent when two people edit the same line offline?"),
    ("hard", "What happens to gradient flow in a 50-layer plain network without residual connections?"),
    ("hard", "Is it safe to use read-committed isolation for an inventory reservation system?"),
    ("hard", "How should I shard a Postgres table that has hot tenants and cold tenants?"),
    ("hard", "What breaks if I put a semantic cache in front of a RAG system whose documents update hourly?"),
    ("hard", "Should our startup self-host Llama or pay per token for 2 million requests a month?"),
    ("hard", "How does speculative decoding speed up inference without changing the output distribution?"),
    ("hard", "Which is better for a small labelled dataset of 800 legal clauses: few-shot prompting or a LoRA adapter?"),
    ("hard", "How can attention be computed in less memory than the sequence length squared?"),
    ("hard", "What would you change in a microservice that times out only under peak load?"),
    ("hard", "Write a SQL query that finds each customer's second most recent order, handling ties."),
    ("hard", "How do I make retries idempotent for a payment webhook handler?"),
    ("simple", "Explain what a variable is."),
    ("simple", "Why is the sky blue?"),
    ("simple", "Compare tea and coffee caffeine content"),
    ("simple", "What year did Pakistan gain independence?"),
    ("simple", "Translate 'thank you' to French"),
    ("simple", "Convert 5 km to miles"),
    ("simple", "Give me a synonym for happy"),
    ("simple", "Is Python case sensitive?"),
    ("simple", "What does CPU stand for?"),
    ("simple", "List the planets in the solar system"),
    ("simple", "How do I exit vim?"),
    ("simple", "What is the plural of cactus?"),
]


def build_workload(seed: int = 7, repeats: int = 2):
    """Returns a list of (query, intent_id, tier_label) in a realistic interleaved order.

    The first query of each paraphrase group appears before its paraphrases, so
    the cache has a chance to be warm. `repeats` re-asks the whole set in a new
    order, which is where exact-match hits come from (real traffic is repetitive).
    """
    import random

    rng = random.Random(seed)
    items = []
    for gi, (tier, qs) in enumerate(PARAPHRASE_GROUPS):
        for qi, q in enumerate(qs):
            items.append((q, f"p{gi}", tier, qi))
    for gi, qs in enumerate(NEAR_MISS_GROUPS):
        for qi, q in enumerate(qs):
            items.append((q, f"n{gi}_{qi}", "simple", 0))

    first = [it for it in items if it[3] == 0]
    rest = [it for it in items if it[3] != 0]
    rng.shuffle(first)
    rng.shuffle(rest)
    order = first + rest
    for _ in range(repeats - 1):
        again = items[:]
        rng.shuffle(again)
        order += again
    return [(q, intent, tier) for q, intent, tier, _ in order]


def routing_labels():
    out = [(q, tier) for tier, qs in PARAPHRASE_GROUPS for q in qs]
    out += [(q, "simple") for qs in NEAR_MISS_GROUPS for q in qs]
    return out
