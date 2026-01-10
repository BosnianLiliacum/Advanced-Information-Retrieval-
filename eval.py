#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import helix
import torch
from transformers import AutoTokenizer, AutoModel

from insert_data import vectorize_text


# ----------------------------
# Config
# ----------------------------
DB_LOCAL = True
K = 20

SUBREDDIT_FIELD = "subreddit"


# TODO: put more from other datasets as well
# --------------- Queries to evaluate ---------------
EVAL_QUERIES: List[Tuple[str, str]] = [
    # (label, query_text)

    # ollama
    ("ollama", "How do I run Ollama as a background service on Linux?"),
    ("ollama", "Can I run multiple models simultaneously in Ollama?"),
    ("ollama", "How do I import a custom Safetensors model into Ollama?"),
    ("ollama", "What is the difference between num_thread and num_gpu in a Modelfile?"),
    ("ollama", "How do I update Ollama to the latest version without losing models?"),
    ("ollama", "Why is my model offloading to CPU instead of using VRAM?"),
    ("ollama", "How to expose Ollama API to other devices on my local network?"),
    ("ollama", "What are the recommended system requirements for Llama 3 70B?"),
    ("ollama", "How do I set custom system prompts in an Ollama Modelfile?"),
    ("ollama", "How to clear the model cache to free up disk space in Ollama?"),

    # python
    ("python", "How do I handle exceptions and errors properly in Python?"),
    ("python", "What is the difference between __init__ and __call__ in classes?"),
    ("python", "How do I use list comprehensions for cleaner code?"),
    ("python", "What is a decorator and how do I create one in Python?"),
    ("python", "How do I profile a Python script to find performance bottlenecks?"),
    ("python", "What are the advantages of using asyncio over threading?"),
    ("python", "How do I manage dependencies using a requirements.txt file?"),
    ("python", "What is the 'if __name__ == \"__main__\":' block used for?"),
    ("python", "How do I merge two dictionaries in Python 3.9+?"),
    ("python", "Difference between shallow copy and deep copy in Python?"),

    # selfhosted
    ("selfhosted", "What is the best file system for a DIY NAS (ZFS vs BTRFS)?"),
    ("selfhosted", "How do I automate backups for my Docker volumes?"),
    ("selfhosted", "Which hardware is better for a home lab: NUC or old Enterprise server?"),
    ("selfhosted", "How do I set up a local dashboard like Homarr or Dashy?"),
    ("selfhosted", "What are the pros and cons of using Unraid for self-hosting?"),
    ("selfhosted", "How do I monitor my self-hosted services using Uptime Kuma?"),
    ("selfhosted", "Best way to manage media libraries (Plex vs Jellyfin)?"),
    ("selfhosted", "How do I set up an automated 'ARR' stack for media management?"),
    ("selfhosted", "Is it safe to port forward 443 for my home server?"),
    ("selfhosted", "How do I use Cloudflare Tunnels to bypass CGNAT?"),

    # tailscale
    ("Tailscale", "How do I set up Tailscale on a Headless Linux server?"),
    ("Tailscale", "What is Tailscale Funnel and how is it different from a Tailnet?"),
    ("Tailscale", "How do I use ACLs to restrict access between specific devices?"),
    ("Tailscale", "Can I use Tailscale as a global VPN for all my internet traffic?"),
    ("Tailscale", "How do I resolve hostname conflicts in Tailscale?"),
    ("Tailscale", "How do I authenticate a new device using an auth key?"),
    ("Tailscale", "What is the performance overhead of using Tailscale over wireguard?"),
    ("Tailscale", "How do I share a specific node with a friend outside my tailnet?"),
    ("Tailscale", "Does Tailscale work with a double NAT setup?"),
    ("Tailscale", "How do I update the Tailscale client on MacOS?"),

    ("archlinux", "How do I use the Arch Linux RPC API to get package metadata in JSON?"),
    ("archlinux", "What is the rate limit for the Arch Linux official package search?"),
    ("archlinux", "How to scrape the ArchWiki without getting blocked by Cloudflare?"),
    ("archlinux", "Best way to parse a PKGBUILD file to extract dependencies in Python?"),
    ("archlinux", "How do I download the latest 'community.db' for offline package analysis?"),
    ("archlinux", "What are the legal/TOS restrictions for scraping archlinux.org?"),
    ("archlinux", "How to extract maintainer information from the AUR Web RPC?"),
    ("archlinux", "How to parse Arch Linux security advisories (ASAs) programmatically?"),
    ("archlinux", "How to find outdated packages by comparing AUR versions with upstream?"),
    ("archlinux", "How to use Scrapy to crawl Arch Linux mirror list performance data?"),

    ("huggingface", "How do I use HfApi to list all models with a specific tag like 'Llama-3'?"),
    ("huggingface", "How to extract the 'license' field from a Hugging Face model card YAML?"),
    ("huggingface", "What is the most efficient way to scrape dataset download counts via the API?"),
    ("huggingface", "How to download only the 'config.json' from 1000+ models without the weights?"),
    ("huggingface", "How to handle '429 Too Many Requests' when calling the Hugging Face Hub API?"),
    ("huggingface", "How do I scrape the Hugging Face Leaderboard for the latest LLM rankings?"),
    ("huggingface", "How to use the 'huggingface_hub' library to get a user's total likes/downloads?"),
    ("huggingface", "Can I scrape the Hugging Face 'Spaces' metadata to find popular Gradio apps?"),
    ("huggingface", "How to parse the 'README.md' of a model to find hardware requirements?"),
    ("huggingface", "How to get a list of all files in a repository using the Hub REST API?"),

    ("kernel", "How do I scrape kernel.org to find the latest stable vs. mainline versions?"),
    ("kernel", "What is the best way to parse a patch file from the Linux Kernel Mailing List?"),
    ("kernel", "How to use git-web to programmatically download diffs for specific commits?"),
    ("kernel", "How can I scrape the MAINTAINERS file to map drivers to specific developers?"),
    ("kernel", "How to extract changelog summaries from the 'v6.x' release announcements?"),
    ("kernel", "What are the rate limits for git clones or pulls from git.kernel.org?"),
    ("kernel", "How to scrape lore.kernel.org for specific discussion threads on a subsystem?"),
    ("kernel", "How do I automate the download of signed checksum files for kernel tarballs?"),
    ("kernel", "How to parse the Kconfig files to list all available kernel modules?"),
    ("kernel", "How to scrape the Kernel Documentation sphinx pages for specific API references?"),

    ("LangChain", "How to scrape LangChain documentation to build a local RAG knowledge base?"),
    ("LangChain", "How do I extract all supported 'Community' integrations from the LangChain docs?"),
    ("LangChain", "How to parse the LangChain API reference to find deprecated class methods?"),
    ("LangChain", "How can I scrape the LangChain Hub to list the most popular prompt templates?"),
    ("LangChain", "What is the best way to crawl the LangChain blog for new release feature notes?"),
    ("LangChain", "How to scrape LangChain's GitHub issues to track common user 'bug' reports?"),
    ("LangChain", "How to extract code examples from LangChain 'How-to' guides programmatically?"),
    ("LangChain", "How to handle versioned documentation when scraping LangChain (v0.1 vs v0.2)?"),
    ("LangChain", "How to scrape the list of supported LLM providers from the LangChain core library?"),
    ("LangChain", "How to use BeautifulSoup to extract the table of contents from a LangChain doc page?"),

    ("learnprogramming", "How do I use the Reddit API to scrape the top 100 'Getting Started' posts?"),
    ("learnprogramming", "How to extract the most recommended books from the r/learnprogramming wiki?"),
    ("learnprogramming", "How to scrape FAQ sections to identify common hurdles for new developers?"),
    ("learnprogramming", "How to use PRAW to collect 'Project Ideas' threads from the last year?"),
    ("learnprogramming", "How to handle Reddit's pagination when scraping long 'Self-Post' discussions?"),
    ("learnprogramming", "How to scrape and categorize programming language mentions in beginner threads?"),
    ("learnprogramming", "How to extract curated lists of free resources from the r/learnprogramming sidebar?"),
    ("learnprogramming", "What is the best way to clean HTML tags from scraped programming tutorials?"),
    ("learnprogramming", "How to scrape 'Roadmap' threads to create a curriculum for self-learners?"),
    ("learnprogramming", "How to identify and scrape 'Success Story' posts to analyze learning timelines?"),

    ("linux", "How do I scrape the Linux Man Pages (man7.org) for a specific syscall?"),
    ("linux", "How to extract a list of all CLI flags for a command from its man page?"),
    ("linux", "How to scrape the DistroWatch rankings to track Linux distribution popularity?"),
    ("linux", "How to programmatically download all Linux logos from a vector repository?"),
    ("linux", "How to scrape the 'Linux Questions' forums for specific hardware error codes?"),
    ("linux", "What is the best way to parse the Linux Kernel's documentation index?"),
    ("linux", "How to scrape Phoronix for the latest Linux hardware benchmark data?"),
    ("linux", "How to extract command examples from the 'TLDR pages' GitHub repository?"),
    ("linux", "How to scrape Linux hardware compatibility lists (HCL) for specific laptops?"),
    ("linux", "How to parse the '/proc' filesystem documentation to map system variables?"),

    ("MachineLearning", "How to scrape ArXiv to find the most cited ML papers in the last 30 days?"),
    ("MachineLearning", "How to extract 'State of the Art' (SOTA) results from the Papers With Code website?"),
    ("MachineLearning", "How to scrape the r/MachineLearning 'What are you reading?' weekly threads?"),
    ("MachineLearning", "How to use the Semantic Scholar API to scrape author affiliation data?"),
    ("MachineLearning", "How to extract YAML metadata from the official NeurIPS or ICML paper archives?"),
    ("MachineLearning", "How to scrape GitHub to find trending repositories using the 'PyTorch' tag?"),
    ("MachineLearning", "How to extract hyperparameter configurations from README files in ML repos?"),
    ("MachineLearning", "How to scrape the 'OpenReview' platform for peer review scores of new models?"),
    ("MachineLearning", "How to identify and scrape 'Discussions' vs 'Research' flairs on Reddit?"),
    ("MachineLearning", "How to scrape AI newsletters (like Jack Clark’s Import AI) for trend analysis?"),

    ("linuxquestions", "How to scrape LinuxQuestions.org for solved threads regarding GRUB boot errors?"),
    ("linuxquestions", "How to extract hardware specs from 'Why won't my Wi-Fi work?' support threads?"),
    ("linuxquestions", "How to scrape the 'Linux Answers' knowledge base for networking tutorials?"),
    ("linuxquestions", "How to parse thread metadata to find the most active expert contributors?"),
    ("linuxquestions", "How to scrape forum posts to identify recurring issues with kernel updates?"),
    ("linuxquestions", "How to extract bash script snippets from 'Question' vs 'Answer' blocks?"),
    ("linuxquestions", "How to scrape distribution-specific sub-forums for Debian stable release issues?"),
    ("linuxquestions", "How to handle forum pagination when crawling 10+ years of 'Solved' tags?"),
    ("linuxquestions", "How to scrape LinuxQuestions to find advice on reviving 32-bit legacy hardware?"),
    ("linuxquestions", "How to programmatically detect 'abandoned' threads with no replies for data cleaning?"),

    ("linux4noobs", "How to scrape the 'Best Distro for Beginners' threads from the last 6 months?"),
    ("linux4noobs", "How do I extract common troubleshooting steps for NVIDIA drivers on Linux?"),
    ("linux4noobs", "How to scrape Reddit to find the most recommended desktop environments for new users?"),
    ("linux4noobs", "How to extract a list of 'must-have' apps for new Linux users from community posts?"),
    ("linux4noobs", "How to scrape and count the most frequent 'switching from Windows' complaints?"),
    ("linux4noobs", "How to pull the most upvoted 'beginner tips' comments using the PRAW library?"),
    ("linux4noobs", "How to scrape threads about 'Wine vs Proton' to compare gaming advice?"),
    ("linux4noobs", "How to extract partition naming conventions (sda vs nvme) from beginner guides?"),
    ("linux4noobs", "How to scrape the r/linux4noobs wiki for a list of recommended terminal emulators?"),
    ("linux4noobs", "How to identify and scrape threads discussing 'Linux Mint vs Pop!_OS' for newcomers?"),
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
            p = p
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
