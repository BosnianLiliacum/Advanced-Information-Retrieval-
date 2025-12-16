#def func -> [(title, content, URL, score, top_k_comments(comment, score)), ....]

#for dir in "reddit/"
#    for file in dir
#        get_title
#        get_content
#        get_url
#        get_score
#        get_topk_comments

#{sr1: [(title, content, URL, score, top_k_comments(comment, score)), ....], sr2: [(title, content, URL, score, top_k_comments(comment, score)), ....], ...}

# IDEA uphere
#########

#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Comment:
    author: Optional[str]
    text: str
    score: Optional[int]
    created_utc: Optional[str]


@dataclass
class Post:
    subreddit: str
    title: str
    author: Optional[str]
    score: Optional[int]
    url: str
    num_comments: Optional[int]
    created_utc: Optional[str]
    content: str
    comments: List[Comment]

    def to_required_tuple(self, top_k: int) -> Tuple[str, str, str, int, List[Tuple[str, int]]]:
        """
        Return exactly what you asked for:
        (title, content, URL, score, top_k_comments[(comment, score), ...])
        """
        post_score = int(self.score) if self.score is not None else 0

        top = sorted(
            self.comments,
            key=lambda c: (c.score if c.score is not None else -10**9),
            reverse=True,
        )[:top_k]

        top_k_pairs = []
        for c in top:
            c_score = int(c.score) if c.score is not None else 0
            top_k_pairs.append((c.text.strip(), c_score))

        return (self.title.strip(), self.content.strip(), self.url.strip(), post_score, top_k_pairs)


# -----------------------------
# Parsing helpers
# -----------------------------
_INT_RE = re.compile(r"-?\d+")

def _parse_int(line: str) -> Optional[int]:
    m = _INT_RE.search(line)
    return int(m.group(0)) if m else None

def _get_field(lines: List[str], prefix: str) -> Optional[str]:
    # Example: "Post Title: xyz"
    for ln in lines:
        if ln.startswith(prefix):
            return ln[len(prefix):].strip()
    return None

def _split_sections(text: str) -> Dict[str, str]:
    """
    Splits into sections keyed by headers we care about.
    We treat these headers as markers:
      - "Post Content:"
      - "Top " (e.g. "Top 4 comments:")
    Everything above "Post Content:" is metadata lines.
    """
    # Normalize newlines
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    return {"full": t}

def _extract_post_content(full_text: str) -> Tuple[str, str]:
    """
    Returns (metadata_block, content_and_comments_block)
    """
    marker = "\n\nPost Content:\n"
    if marker in full_text:
        meta, rest = full_text.split(marker, 1)
        return meta.strip(), rest.strip()
    # fallback: sometimes "Post Content:" may be directly after newline
    marker2 = "\nPost Content:\n"
    if marker2 in full_text:
        meta, rest = full_text.split(marker2, 1)
        return meta.strip(), rest.strip()
    # if missing, treat all as meta
    return full_text.strip(), ""

def _extract_comments_block(content_and_more: str) -> Tuple[str, str]:
    """
    Returns (content_text, comments_block)
    """
    # Typical marker: "\n\nTop 4 comments:\n"
    m = re.search(r"\n\s*Top\s+\d+\s+comments:\s*\n", content_and_more)
    if not m:
        return content_and_more.strip(), ""
    content = content_and_more[:m.start()].strip()
    comments = content_and_more[m.end():].strip()
    return content, comments

def _parse_comments(comments_block: str) -> List[Comment]:
    if not comments_block:
        return []

    # Split by "Comment X:" blocks
    # Example:
    # Comment 1:
    #   Author: ...
    #   Comment: ...
    #   Score: ...
    #   Created UTC: ...
    parts = re.split(r"\n\s*Comment\s+\d+\s*:\s*\n", "\n" + comments_block)
    # first element before first comment will be junk
    parts = [p.strip() for p in parts if p.strip()]

    comments: List[Comment] = []
    for p in parts:
        lines = p.split("\n")

        author = None
        score = None
        created = None

        # Comment text can span multiple lines; it starts after "Comment:" (or "  Comment:")
        comment_text_lines: List[str] = []
        in_comment_text = False

        for ln in lines:
            s = ln.strip()

            if s.startswith("Author:"):
                author = s[len("Author:"):].strip()
                in_comment_text = False
                continue

            if s.startswith("Score:"):
                score = _parse_int(s)
                in_comment_text = False
                continue

            if s.startswith("Created UTC:"):
                created = s[len("Created UTC:"):].strip()
                in_comment_text = False
                continue

            if s.startswith("Comment:"):
                in_comment_text = True
                comment_text_lines.append(s[len("Comment:"):].lstrip())
                continue

            # Keep collecting multi-line comment text until we hit another known field
            if in_comment_text:
                comment_text_lines.append(ln)

        text = "\n".join(comment_text_lines).strip()
        if text:  # only keep real comments
            comments.append(Comment(author=author, text=text, score=score, created_utc=created))

    return comments

def parse_post_file(path: Path) -> Post:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta_block, rest = _extract_post_content(text)

    meta_lines = [ln.strip() for ln in meta_block.split("\n") if ln.strip()]

    title = _get_field(meta_lines, "Post Title:") or ""
    author = _get_field(meta_lines, "Author:")
    score = _parse_int(_get_field(meta_lines, "Score:") or "")
    url = _get_field(meta_lines, "URL:") or ""
    num_comments = _parse_int(_get_field(meta_lines, "Number of comments:") or "")
    created = _get_field(meta_lines, "Created UTC:")

    content_text, comments_block = _extract_comments_block(rest)
    comments = _parse_comments(comments_block)

    # subreddit name is the folder name under reddit/scrapes/
    subreddit = path.parent.name.replace("scrape_", "", 1)

    return Post(
        subreddit=subreddit,
        title=title,
        author=author,
        score=score,
        url=url,
        num_comments=num_comments,
        created_utc=created,
        content=content_text,
        comments=comments,
    )


# -----------------------------
# Main API you asked for
# -----------------------------
def load_all_posts(
    root: str | Path = "reddit/scrapes",
    top_k_comments: int = 4,
) -> List[Tuple[str, str, str, str, int, List[Tuple[str, int]]]]:
    """
    Returns:
      (subreddit, title, content, url, score, [(comment_text, comment_score), ...])
    """
    # Make root path robust regardless of where you run `python preprocess.py` from
    script_dir = Path(__file__).resolve().parent
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (script_dir / root_path).resolve()

    print(f"[INFO] Scanning root: {root_path}")
    if not root_path.exists():
        print(f"[ERROR] Root folder does not exist: {root_path}")
        print("[HINT] Check your repo structure. Expected something like: <repo>/reddit/scrapes/...")
        return []

    # Try multiple patterns in case your filenames differ
    patterns = [
        "**/post_*.txt",
        "**/post*.txt",
        "**/*.txt",
        "**/*.md",
    ]

    files = []
    for pat in patterns:
        matches = list(root_path.glob(pat))
        if matches:
            print(f"[INFO] Pattern '{pat}' matched {len(matches)} files.")
            files = matches
            break

    if not files:
        # show what folders exist to help you spot the mismatch
        subs = [p for p in root_path.iterdir() if p.is_dir()]
        print("[WARN] No files found with expected patterns.")
        print("[INFO] Subfolders under root:")
        for p in subs[:30]:
            print("  -", p.name)
        print("[HINT] Your files might not be .txt or might not start with 'post'.")
        return []

    # Show a few found files
    print("[INFO] Example files found:")
    for f in sorted(files)[:5]:
        print("  -", f)

    out = []
    for f in sorted(files):
        try:
            post = parse_post_file(f)  # uses your parser
            title, content, url, score, top = post.to_required_tuple(top_k_comments)
            out.append((post.subreddit, title, content, url, score, top))
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")

    return out


# -----------------------------
# Optional: quick CLI test
# -----------------------------
if __name__ == "__main__":
    posts = load_all_posts("datasets/scrapes", top_k_comments=4)
    print(f"Loaded {len(posts)} posts.")
    # Show one example
    if posts:
        sr, title, content, url, score, top = posts[0]
        print("\n--- Example ---")
        print("Subreddit:", sr)
        print("Title:", title)
        print("URL:", url)
        print("Score:", score)
        print("Top comments:", top)
        print(len(posts))
