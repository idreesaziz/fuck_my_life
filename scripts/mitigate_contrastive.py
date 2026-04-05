#!/usr/bin/env python3
"""
Contrastive Anchor Prompting Mitigation
=========================================
For each of 200 randomly sampled texts, score with TWO prompt orders:
  Order A: "Rate this text 0-10" (standard)
  Order B: "Rate this text 0-10. A score of 0 means <worst> and 10 means <best>."
           + reversed anchor ("A score of 10 means <best> and 0 means <worst>.")

Uses GPT-5-mini and Gemini-3-flash (API models only — local models too slow).
Implements checkpoint/resume logic; 200 paired calls per model.

Outputs:
  output/mitigations/intermediate/contrastive_raw.csv
  output/mitigations/results/contrastive_delta.csv
  output/mitigations/figures/contrastive_delta_hist.png

Usage:  python scripts/mitigate_contrastive.py
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import (
    RANDOM_STATE, LEVELS, proxy_ground_truth, compute_compression_ratio,
    pairwise_accuracy,
)
from llm_scoring import parse_score

from dotenv import load_dotenv
load_dotenv()

# ── Paths ───────────────────────────────────────────────────────
INT_DIR = ROOT / "output" / "mitigations" / "intermediate"
RES_DIR = ROOT / "output" / "mitigations" / "results"
FIG_DIR = ROOT / "output" / "mitigations" / "figures"
INT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)

N_SAMPLES = 200

# ── System prompts (instruction only — user content is text only, like the
#    original benchmark which uses SYSTEM_PROMPT + user_prompt=text) ──────────
SYSTEM_STANDARD = "Rate the quality of the following text from 0 to 10. Respond with ONLY the number."

SYSTEM_ANCHOR_A = (
    "Rate the quality of the following text from 0 to 10. "
    "A score of 0 means the text is completely unreadable, and 10 means it is "
    "flawless professional writing. Respond with ONLY the number."
)

SYSTEM_ANCHOR_B = (
    "Rate the quality of the following text from 0 to 10. "
    "A score of 10 means the text is flawless professional writing, and 0 means "
    "it is completely unreadable. Respond with ONLY the number."
)


# ── API Models config ──────────────────────────────────────────
API_MODELS = [
    {
        "name": "gpt-5-mini",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    {
        "name": "gemini-3-flash",
        "provider": "google",
        "model_id": "gemini-3-flash-preview",
        "api_key_env": "GOOGLE_API_KEY",
    },
]


# ═══════════════════════════════════════════════════════════════
# LLM CLIENTS  (created once, reused across all calls)
# ═══════════════════════════════════════════════════════════════
_CLIENTS: dict = {}


def _get_client(provider: str, api_key_env: str):
    """Return a cached client for the given provider."""
    if provider not in _CLIENTS:
        api_key = os.environ.get(api_key_env)
        if provider == "google":
            # Store just the api_key; calls go via requests (genuine timeout)
            _CLIENTS[provider] = api_key
        else:
            from openai import OpenAI
            _CLIENTS[provider] = OpenAI(api_key=api_key, timeout=30)
    return _CLIENTS[provider]


CALL_TIMEOUT = 30  # seconds — enforced at socket level for Gemini, SDK level for OpenAI

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={key}"
)


def _raw_llm_call(provider: str, model_id: str, system_prompt: str,
                  user_text: str, client) -> str:
    """Inner call — matches original benchmark settings per provider.

    Google: system_instruction, temperature=0, thinking_level=minimal
    OpenAI: system message + user message, temperature=0, max_tokens=10
    """
    if provider == "google":
        api_key = client  # stored as plain string
        url = _GEMINI_URL.format(model=model_id, key=api_key)
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": 0.0,
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        }
        resp = requests.post(url, json=body, timeout=(10, CALL_TIMEOUT))
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content or ""


def call_llm(provider: str, model_id: str, system_prompt: str,
             user_text: str, api_key_env: str = "OPENAI_API_KEY") -> int | None:
    """Single synchronous LLM call with retry. Gemini uses requests (real socket timeout)."""
    client = _get_client(provider, api_key_env)

    for attempt in range(3):
        try:
            raw = _raw_llm_call(provider, model_id, system_prompt, user_text, client)
            score = parse_score(raw)
            if score is None:
                print(f"    [Parse fail] raw={raw!r}", flush=True)
            return score

        except requests.exceptions.Timeout:
            print(f"    [Timeout {CALL_TIMEOUT}s] attempt {attempt+1}/3", flush=True)
            time.sleep(2)
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["429", "rate", "limit", "quota"]):
                wait = 2 ** attempt * 5
                print(f"    [Retry {attempt+1}] Rate limit, waiting {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"    [Error] {str(e)[:120]}", flush=True)
                return None
    return None


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Load degraded samples
    samples = json.loads(
        (ROOT / "data/degraded/degraded_samples.json").read_text(encoding="utf-8"))

    # Stratified sample: pick N_SAMPLES across levels/axes
    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(len(samples), size=min(N_SAMPLES, len(samples)), replace=False)
    subset = [samples[i] for i in sorted(indices)]
    print(f"Selected {len(subset)} samples for contrastive prompting")

    # Load checkpoint if exists
    ckpt_path = INT_DIR / "contrastive_raw.csv"
    done = set()
    if ckpt_path.exists():
        existing = pd.read_csv(ckpt_path)
        done = set(zip(existing["sample_id"], existing["model"], existing["prompt_order"]))
        print(f"  Resuming: {len(done)} entries already done")

    results = []
    if ckpt_path.exists():
        results = pd.read_csv(ckpt_path).to_dict("records")

    for model_cfg in API_MODELS:
        model_name = model_cfg["name"]
        provider = model_cfg["provider"]
        model_id = model_cfg["model_id"]
        api_key_env = model_cfg["api_key_env"]

        if not os.environ.get(api_key_env):
            print(f"  [SKIP] {model_name}: {api_key_env} not set")
            continue

        print(f"\n  Scoring with {model_name} …", flush=True)
        n_new = 0
        n_fail = 0
        total_todo = sum(
            1 for s in subset
            for order in ("standard", "anchor_A", "anchor_B")
            if (s["id"], model_name, order) not in done
        )
        print(f"    {total_todo} calls remaining", flush=True)

        for s in subset:
            sid = s["id"]
            text = s["degraded_text"]

            for order, sys_prompt in [("standard", SYSTEM_STANDARD),
                                      ("anchor_A", SYSTEM_ANCHOR_A),
                                      ("anchor_B", SYSTEM_ANCHOR_B)]:
                key = (sid, model_name, order)
                if key in done:
                    continue

                score = call_llm(provider, model_id, sys_prompt, text, api_key_env)

                if score is None:
                    n_fail += 1
                    continue

                results.append({
                    "sample_id": sid,
                    "model": model_name,
                    "prompt_order": order,
                    "score": score,
                    "axis": s["axis"],
                    "level": s["level"],
                    "category": s.get("category", "unknown"),
                    "article": s["source_title"],
                })
                done.add(key)
                n_new += 1

                # Checkpoint every 20 calls
                if n_new % 20 == 0:
                    pd.DataFrame(results).to_csv(ckpt_path, index=False)
                    print(f"    checkpoint {n_new}/{total_todo}  (fails={n_fail})", flush=True)

                # Small delay to avoid rate limits
                time.sleep(0.3)

        # Final save for this model
        pd.DataFrame(results).to_csv(ckpt_path, index=False)
        print(f"    {model_name}: {n_new} new calls done, {n_fail} failures", flush=True)

    if not results:
        print("No results. Set OPENAI_API_KEY / GOOGLE_API_KEY to run contrastive prompting.")
        return

    print(f"\n  Saved raw → {ckpt_path}")

    # ── Compute deltas ──────────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.dropna(subset=["score"])

    # Pivot: for each (sample_id, model), get scores for each prompt_order
    pivot = df.pivot_table(index=["sample_id", "model", "axis", "level"],
                           columns="prompt_order", values="score",
                           aggfunc="first").reset_index()

    delta_rows = []
    for _, row in pivot.iterrows():
        std = row.get("standard")
        anchor_a = row.get("anchor_A")
        anchor_b = row.get("anchor_B")

        if pd.notna(std) and pd.notna(anchor_a):
            delta_rows.append({
                "sample_id": row["sample_id"],
                "model": row["model"],
                "axis": row["axis"],
                "level": row["level"],
                "score_standard": std,
                "score_anchor_A": anchor_a,
                "score_anchor_B": anchor_b if pd.notna(anchor_b) else None,
                "delta_A_minus_std": anchor_a - std,
                "delta_B_minus_std": (anchor_b - std) if pd.notna(anchor_b) else None,
            })

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(RES_DIR / "contrastive_delta.csv", index=False)
    print(f"  Saved deltas → {RES_DIR / 'contrastive_delta.csv'}")

    # ── Figure: delta histogram ─────────────────────────────────
    models = sorted(delta_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0][idx]
        msub = delta_df[delta_df["model"] == model]

        if "delta_A_minus_std" in msub.columns:
            ax.hist(msub["delta_A_minus_std"].dropna(), bins=21,
                    alpha=0.6, color="#1f77b4", label="Anchor A − Std", edgecolor="white")
        if "delta_B_minus_std" in msub.columns:
            ax.hist(msub["delta_B_minus_std"].dropna(), bins=21,
                    alpha=0.6, color="#ff7f0e", label="Anchor B − Std", edgecolor="white")

        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("Score Delta (anchor − standard)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    fig.suptitle("Contrastive Prompting: Score Deltas", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "contrastive_delta_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'contrastive_delta_hist.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
