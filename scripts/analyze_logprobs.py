#!/usr/bin/env python3
"""
Log-Probability Feature Extraction & Visualisation
====================================================
Works on models that have `score_probs` (dict of 11 probabilities, keys "0"–"10").

Extracts per-sample features:
  entropy, p_ceiling (p(9)+p(10)), expected_score, argmax_score,
  expected_minus_argmax, kl_vs_uniform, top2_gap, score_probs array.

Outputs:
  output/mitigations/intermediate/logprob_features.csv
  output/mitigations/figures/logprob_entropy.png
  output/mitigations/figures/logprob_p_ceiling.png
  output/mitigations/figures/logprob_expected_vs_argmax.png
  output/mitigations/figures/logprob_heatmap.png
  output/mitigations/figures/logprob_argmax_deviation.png

Usage:  python scripts/analyze_logprobs.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import (
    AXES_ORDER, LEVELS, RANDOM_STATE, load_scores, proxy_ground_truth,
)

# ── Output dirs ─────────────────────────────────────────────────
INT_DIR = ROOT / "output" / "mitigations" / "intermediate"
FIG_DIR = ROOT / "output" / "mitigations" / "figures"
INT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_logprob_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract logprob features from rows that have score_probs."""
    rows = []
    for _, r in df.iterrows():
        sp = r.get("score_probs")
        if not sp or not isinstance(sp, dict):
            continue

        probs = np.array([float(sp.get(str(i), sp.get(i, 0.0))) for i in range(11)])
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total  # re-normalise

        # Entropy (base 2)
        entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

        # Ceiling probability
        p_ceiling = float(probs[9] + probs[10])

        # Expected score
        expected = float(np.dot(np.arange(11), probs))

        # Argmax score
        argmax = int(np.argmax(probs))

        # KL divergence vs uniform
        uniform = np.full(11, 1 / 11)
        kl = float(sp_stats.entropy(probs, uniform, base=2))

        # Top-2 gap
        sorted_probs = np.sort(probs)[::-1]
        top2_gap = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0

        rows.append({
            "sample_id": r["sample_id"],
            "model": r["model"],
            "axis": r["axis"],
            "level": r["level"],
            "article": r["article"],
            "category": r["category"],
            "score": r["score"],
            "entropy": entropy,
            "p_ceiling": p_ceiling,
            "expected_score": expected,
            "argmax_score": argmax,
            "expected_minus_argmax": expected - argmax,
            "kl_vs_uniform": kl,
            "top2_gap": top2_gap,
            **{f"p_{i}": float(probs[i]) for i in range(11)},
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════

def fig_entropy(lp: pd.DataFrame):
    """Entropy by degradation level, per model."""
    models = sorted(lp["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = lp[lp["model"] == model]
        sns.boxplot(data=sub, x="level", y="entropy", ax=ax,
                    palette="viridis", width=0.6)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Entropy (bits)" if idx % ncols == 0 else "")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Score-Probability Entropy by Degradation Level", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logprob_entropy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'logprob_entropy.png'}")


def fig_p_ceiling(lp: pd.DataFrame):
    """P(ceiling) = P(9)+P(10) by degradation level."""
    models = sorted(lp["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = lp[lp["model"] == model]
        grouped = sub.groupby("level")["p_ceiling"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(grouped["level"], grouped["mean"],
                    yerr=grouped["sem"] * 1.96,
                    marker="o", capsize=3, linewidth=2)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("P(ceiling)" if idx % ncols == 0 else "")
        ax.set_ylim(-0.05, 1.05)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Ceiling Probability P(9)+P(10) by Degradation Level", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logprob_p_ceiling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'logprob_p_ceiling.png'}")


def fig_expected_vs_argmax(lp: pd.DataFrame):
    """Scatter: expected score vs argmax score per model."""
    models = sorted(lp["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = lp[lp["model"] == model]
        ax.scatter(sub["argmax_score"], sub["expected_score"],
                   alpha=0.1, s=8, color="#1f77b4")
        ax.plot([0, 10], [0, 10], "--k", alpha=0.4, linewidth=1)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Argmax Score")
        ax.set_ylabel("Expected Score" if idx % ncols == 0 else "")
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Expected Score vs Argmax Score", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logprob_expected_vs_argmax.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'logprob_expected_vs_argmax.png'}")


def fig_heatmap(lp: pd.DataFrame):
    """Per-model heatmap: mean P(score) × degradation level."""
    models = sorted(lp["model"].unique())
    n = len(models)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = lp[lp["model"] == model]
        prob_cols = [f"p_{i}" for i in range(11)]
        heat = sub.groupby("level")[prob_cols].mean()
        heat.columns = [str(i) for i in range(11)]
        sns.heatmap(heat, ax=ax, cmap="YlOrRd", vmin=0, annot=True,
                    fmt=".2f", cbar_kws={"shrink": 0.8})
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Score Token")
        ax.set_ylabel("Degradation Level")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Mean Score-Token Probability Heatmaps", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logprob_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'logprob_heatmap.png'}")


def fig_argmax_deviation(lp: pd.DataFrame):
    """Histogram of (expected − argmax) per model."""
    models = sorted(lp["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = lp[lp["model"] == model]
        ax.hist(sub["expected_minus_argmax"], bins=40, color="#2ca02c",
                alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Expected − Argmax")
        ax.set_ylabel("Count" if idx % ncols == 0 else "")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Distribution of Expected − Argmax Score", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logprob_argmax_deviation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'logprob_argmax_deviation.png'}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading scores …")
    df = load_scores(ROOT)

    # Filter to models with score_probs
    has_probs = df["score_probs"].apply(lambda x: isinstance(x, dict) and len(x) > 0)
    df_lp = df[has_probs].copy()
    print(f"  {len(df_lp)} rows with logprobs from "
          f"{sorted(df_lp['model'].unique())}")

    if df_lp.empty:
        print("No logprob data found. Exiting.")
        return

    print("Extracting features …")
    lp = extract_logprob_features(df_lp)
    print(f"  {len(lp)} feature rows extracted")

    # Save intermediate
    save_cols = [c for c in lp.columns if not c.startswith("p_") or c in
                 [f"p_{i}" for i in range(11)]]
    lp.to_csv(INT_DIR / "logprob_features.csv", index=False)
    print(f"  Saved → {INT_DIR / 'logprob_features.csv'}")

    # Figures
    print("Generating figures …")
    fig_entropy(lp)
    fig_p_ceiling(lp)
    fig_expected_vs_argmax(lp)
    fig_heatmap(lp)
    fig_argmax_deviation(lp)

    print("\nDone.")


if __name__ == "__main__":
    main()
