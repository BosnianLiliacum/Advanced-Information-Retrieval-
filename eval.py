#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import helix
import torch
from transformers import AutoTokenizer, AutoModel

# You already have this helper (good: keeps embedding consistent with indexing)
from insert_data import vectorize_text


# ----------------------------
# Config
# ----------------------------
DB_LOCAL = True
K = 20

# CHANGE THIS to match your DB schema
# inspect one result and set this correctly:
# e.g. "subreddit", "source", "dataset", "community"
SUBREDDIT_FIELD = "subreddit"


# --------------- Queries to evaluate ---------------
# Put ~20 per subreddit (you can extend this list)
EVAL_QUERIES: List[Tuple[str, str]] = [
    # (label, query_text)

    # ollama
    ("ollama", "Why is Ollama slower when loading some GGUF models compared to pulling models directly?"),
    ("ollama", "How do I change the context length in Ollama?"),
    ("ollama", "Does quantization level affect token generation speed in Ollama?"),
    ("ollama", "How do I use Ollama with a GPU on Windows?"),
    ("ollama", "Why does Ollama use so much RAM when running larger models?"),

    # python
    ("python", "What is the difference between a list and a tuple in Python?"),
    ("python", "How do I create a virtual environment on Windows for Python?"),
    ("python", "Best way to parse JSON files efficiently in Python?"),
    ("python", "How does Python's GIL affect multithreading performance?"),
    ("python", "How do I read many text files in a directory in Python?"),

    # selfhosted
    ("selfhosted", "What’s a good self-hosted alternative to Google Photos?"),
    ("selfhosted", "How do I secure a self-hosted service exposed to the internet?"),
    ("selfhosted", "What’s the easiest way to self-host a password manager?"),
    ("selfhosted", "Should I run services in Docker or on bare metal for self-hosting?"),
    ("selfhosted", "How do I set up reverse proxy for multiple self-hosted apps?"),

    # tailscale
    ("tailscale", "Should I run Tailscale in Docker or install it on the host?"),
    ("tailscale", "How do I use a device as an exit node in Tailscale?"),
    ("tailscale", "Why can’t my Tailscale clients reach each other on the tailnet?"),
    ("tailscale", "How do I enable subnet routing with Tailscale?"),
    ("tailscale", "How do I fix DNS issues in Tailscale MagicDNS?"),
]


       # print(f"{lab:10s}  n={stats['n']:2d}  avg_recall@k={stats['avg_recall@k']:.3f}  hit@k={stats['hit_rate@k']:.3f}")
class search_posts_vec(helix.Query):
    def __init__(self, query_vec: List[float], k: int):
        super().__init__()
        self.query_vec = query_vec
        self.k = k

    def query(self) -> List[helix.Payload]:
        return [{"query": self.query_vec, "k": self.k}]

    def response(self, response):
        return response


# ----------------------------
# Metrics
# ----------------------------
@dataclass
class QueryResult:
    label: str
    query: str
    retrieved_labels: List[str]
    recall_at_k: float
    hit_at_k: int


def extract_label(post_obj: Dict[str, Any], field: str) -> Optional[str]:
    # Robust extraction (handles nested dicts if needed)
    if field in post_obj:
        return str(post_obj[field])
    # common fallbacks:
    for alt in ("sr", "sub", "source", "dataset"):
        if alt in post_obj:
            return str(post_obj[alt])
    return None


def compute_recall(retrieved: List[str], label: str) -> float:
    if not retrieved:
        return 0.0
    correct = sum(1 for r in retrieved if r == label)
    return correct / len(retrieved)


def run_eval(db: helix.Client, queries: List[Tuple[str, str]], k: int) -> Tuple[List[QueryResult], Dict[str, Any]]:
    results: List[QueryResult] = []

    for label, text in queries:
        vec = vectorize_text(text)
        res = db.query(search_posts_vec(vec, k))

        # Helix response shape (from your print):
        # res[0]["posts"] -> list of post dicts
        posts = res[0].get("posts", []) if res else []

        retrieved_labels = []
        for p in posts:
            lab = extract_label(p, SUBREDDIT_FIELD)
            if lab is not None:
                retrieved_labels.append(lab)

        # If labels are missing, keep placeholders (so you notice)
        if not retrieved_labels and posts:
            retrieved_labels = ["<missing_label>"] * len(posts)

        recall_k = compute_recall(retrieved_labels, label)
        hit_k = 1 if (label in retrieved_labels) else 0

        results.append(QueryResult(
            label=label,
            query=text,
            retrieved_labels=retrieved_labels,
            recall_at_k=recall_k,
            hit_at_k=hit_k
        ))

    # aggregate
    by_label = defaultdict(list)
    for r in results:
        by_label[r.label].append(r)

    summary = {
        "k": k,
        "total_queries": len(results),
        "overall": {
            "avg_recall@k": sum(r.recall_at_k for r in results) / max(1, len(results)),
            "hit_rate@k": sum(r.hit_at_k for r in results) / max(1, len(results)),
        },
        "per_label": {}
    }

    for lab, items in by_label.items():
        summary["per_label"][lab] = {
            "n": len(items),
            "avg_recall@k": sum(x.recall_at_k for x in items) / len(items),
            "hit_rate@k": sum(x.hit_at_k for x in items) / len(items),
        }

    return results, summary


def main():
    db = helix.Client(local=True, verbose=True)

    results, summary = run_eval(db, EVAL_QUERIES, K)

    print("\n====================")
    print("RAG Retrieval Evaluation")
    print("====================")
    print(f"k = {summary['k']}")
    print(f"total queries = {summary['total_queries']}")
    print(f"overall avg recall@k = {summary['overall']['avg_recall@k']:.3f}")
    print(f"overall hit rate@k   = {summary['overall']['hit_rate@k']:.3f}")

    print("\n--- Per subreddit ---")
    for lab, stats in summary["per_label"].items():
        print(
            f"{lab:10s}  "
            f"n={stats['n']:2d}  "
            f"avg_recall@k={stats['avg_recall@k']:.3f}  "
            f"hit@k={stats['hit_rate@k']:.3f}"
        )

    # Optional: worst queries
    worst = sorted(results, key=lambda r: r.recall_at_k)[:5]
    print("\n--- Worst 5 queries (by recall@k) ---")
    for w in worst:
        print(f"\n[{w.label}] recall@{K}={w.recall_at_k:.3f} hit={w.hit_at_k}")
        print("Q:", w.query)
        print("Top labels:", w.retrieved_labels[:10])



if __name__ == "__main__":
    main()
