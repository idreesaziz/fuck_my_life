"""
Step 1 — Corpus Fetcher
Retrieves Wikipedia articles to serve as Band 9 baseline texts.
"""

import json
import os
from pathlib import Path

import wikipediaapi
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_articles(config: dict) -> list[dict]:
    """Fetch Wikipedia featured articles specified in config. Returns list of
    {title, category, text} dicts with text truncated to max_chars."""

    wiki = wikipediaapi.Wikipedia(
        user_agent="CondemnedToB/1.0 (research project)",
        language="en",
    )

    corpus_cfg = config["corpus"]
    max_chars = corpus_cfg["max_chars"]
    articles = []
    failed = []

    for entry in corpus_cfg["articles"]:
        # Config entries are {title, category} dicts
        title = entry["title"] if isinstance(entry, dict) else entry
        category = entry.get("category", "uncategorized") if isinstance(entry, dict) else "uncategorized"

        page = wiki.page(title)
        if not page.exists():
            print(f"  [WARN] Article not found: {title}")
            failed.append(title)
            continue

        text = page.text if max_chars <= 0 else page.text[:max_chars]
        if len(text) < 500:
            print(f"  [WARN] Article too short ({len(text)} chars): {title}")
            failed.append(title)
            continue

        articles.append({"title": title, "category": category, "text": text})
        print(f"  [{len(articles):3d}/150] {title} ({len(text)} chars) [{category}]")

    if failed:
        print(f"\n  [SUMMARY] {len(failed)} articles failed: {failed}")
    print(f"  [SUMMARY] {len(articles)} articles fetched successfully")

    return articles


def save_corpus(articles: list[dict], output_dir: str) -> Path:
    """Save fetched articles to a JSON file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "corpus.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(articles)} articles to {path}")
    return path


def load_corpus(output_dir: str) -> list[dict]:
    """Load previously saved corpus."""
    path = Path(output_dir) / "corpus.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(config: dict) -> list[dict]:
    """Fetch and save corpus, or load if already present."""
    output_dir = config["corpus"]["output_dir"]
    corpus_file = Path(output_dir) / "corpus.json"

    if corpus_file.exists():
        print("[Corpus] Loading existing corpus...")
        return load_corpus(output_dir)

    print("[Corpus] Fetching Wikipedia articles...")
    articles = fetch_articles(config)
    save_corpus(articles, output_dir)
    return articles


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
