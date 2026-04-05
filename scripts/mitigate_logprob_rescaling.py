#!/usr/bin/env python3
"""
Log-Prob Rescaling Mitigation
==============================
Two methods:
  1. Expected Score   — E[s] = Σ i·p(i) replaces argmax
  2. Asymmetry Fix    — penalise upward skew: s_adj = E[s] − α·skewness
     α tuned by GridSearchCV (5-fold) to minimise RMSE vs proxy GT.

Reads logprob_features.csv from analyze_logprobs.py.

Outputs:
  output/mitigations/results/logprob_rescaling.csv
  output/mitigations/figures/logprob_rescaling_kde.png

Usage:  python scripts/mitigate_logprob_rescaling.py
"""

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
    AXES_ORDER, LEVELS, RANDOM_STATE, proxy_ground_truth,
    compute_compression_ratio, pairwise_accuracy,
)

# ── Paths ───────────────────────────────────────────────────────
INT_DIR = ROOT / "output" / "mitigations" / "intermediate"
RES_DIR = ROOT / "output" / "mitigations" / "results"
FIG_DIR = ROOT / "output" / "mitigations" / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_skewness(row: pd.Series) -> float:
    """Compute skewness of the score probability distribution."""
    probs = np.array([row[f"p_{i}"] for i in range(11)])
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    mean = np.dot(np.arange(11), probs)
    var = np.dot((np.arange(11) - mean) ** 2, probs)
    if var <= 0:
        return 0.0
    std = np.sqrt(var)
    skew = np.dot((np.arange(11) - mean) ** 3, probs) / (std ** 3)
    return float(skew)


def eval_scores(scores: np.ndarray, levels: np.ndarray) -> dict:
    target = proxy_ground_truth(levels)
    wd = float(sp_stats.wasserstein_distance(scores, target))
    cr = compute_compression_ratio(scores)
    rho, rho_p = sp_stats.spearmanr(scores, target)
    slope = sp_stats.linregress(levels, scores).slope
    pa = pairwise_accuracy(scores, target)
    return {
        "wasserstein": round(wd, 4),
        "compression_ratio": round(cr, 4),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rho_p),
        "dose_response_slope": round(float(slope), 4),
        "pairwise_accuracy": round(pa, 4),
    }


def tune_alpha(expected: np.ndarray, skewness: np.ndarray,
               levels: np.ndarray, n_folds: int = 5) -> float:
    """Grid-search α in [0, 3] step 0.1 to minimise RMSE vs proxy GT (KFold)."""
    from sklearn.model_selection import KFold
    target = proxy_ground_truth(levels)
    alphas = np.arange(0.0, 3.05, 0.1)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    best_alpha, best_rmse = 0.0, np.inf
    for a in alphas:
        rmses = []
        for _, val_idx in kf.split(expected):
            adj = np.clip(expected[val_idx] - a * skewness[val_idx], 0, 10)
            rmse = np.sqrt(np.mean((adj - target[val_idx]) ** 2))
            rmses.append(rmse)
        mean_rmse = np.mean(rmses)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_alpha = a

    return float(best_alpha)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    feat_path = INT_DIR / "logprob_features.csv"
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found. Run analyze_logprobs.py first.")
        sys.exit(1)

    print("Loading logprob features …")
    lp = pd.read_csv(feat_path)
    models = sorted(lp["model"].unique())
    print(f"  {len(lp)} rows, {len(models)} models")

    all_results = []

    for model in models:
        sub = lp[lp["model"] == model].copy()
        levels = sub["level"].values
        raw_scores = sub["score"].values.astype(float)
        expected = sub["expected_score"].values

        # Compute skewness per row
        prob_cols = [f"p_{i}" for i in range(11)]
        if not all(c in sub.columns for c in prob_cols):
            print(f"  {model}: missing p_* columns, skipping")
            continue

        skewness = sub.apply(compute_skewness, axis=1).values

        # Tune alpha
        alpha = tune_alpha(expected, skewness, levels)
        adjusted = np.clip(expected - alpha * skewness, 0, 10)

        # Metrics
        m_raw = eval_scores(raw_scores, levels)
        m_raw.update({"model": model, "method": "raw_argmax"})

        m_exp = eval_scores(expected, levels)
        m_exp.update({"model": model, "method": "expected_score"})

        m_adj = eval_scores(adjusted, levels)
        m_adj.update({"model": model, "method": f"asymmetry_fix_alpha={alpha:.1f}"})

        all_results.extend([m_raw, m_exp, m_adj])

        print(f"  {model:30s}  α={alpha:.1f}  "
              f"CR raw={m_raw['compression_ratio']:.3f}  "
              f"exp={m_exp['compression_ratio']:.3f}  "
              f"adj={m_adj['compression_ratio']:.3f}")

    # Save
    cols = ["model", "method", "wasserstein", "compression_ratio",
            "spearman_rho", "spearman_p", "dose_response_slope", "pairwise_accuracy"]
    pd.DataFrame(all_results)[cols].to_csv(
        RES_DIR / "logprob_rescaling.csv", index=False)
    print(f"\n  Saved → {RES_DIR / 'logprob_rescaling.csv'}")

    # ── Figure: KDE ─────────────────────────────────────────────
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = lp[lp["model"] == model]
        raw = sub["score"].values.astype(float)
        exp = sub["expected_score"].values

        # Recompute adjusted for this model
        skew_vals = sub.apply(compute_skewness, axis=1).values
        alpha_m = tune_alpha(exp, skew_vals, sub["level"].values)
        adj = np.clip(exp - alpha_m * skew_vals, 0, 10)

        sns.kdeplot(raw, ax=ax, label="Raw (argmax)", color="#d62728", linewidth=1.5)
        sns.kdeplot(exp, ax=ax, label="E[score]", color="#2ca02c", linewidth=1.5)
        sns.kdeplot(adj, ax=ax, label=f"Asym (α={alpha_m:.1f})",
                    color="#1f77b4", linewidth=1.5)

        ax.set_title(model, fontsize=10)
        ax.set_xlim(-0.5, 10.5)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Logprob Rescaling: Score Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logprob_rescaling_kde.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'logprob_rescaling_kde.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
