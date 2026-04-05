#!/usr/bin/env python3
"""
Quantile-Normalisation Mitigation
==================================
Two variants:
  • Uniform target  — map quantile ranks to Uniform[0, 10]
  • Beta target     — map quantile ranks to Beta(2, 2) scaled to [0, 10]

For every model, computes:
  Wasserstein distance, compression ratio, Spearman ρ, dose-response slope,
  pairwise accuracy (all vs proxy ground truth).

Outputs:
  output/mitigations/results/quantile_uniform.csv
  output/mitigations/results/quantile_beta.csv
  output/mitigations/figures/quantile_kde.png

Usage:  python scripts/mitigate_quantile.py
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
    AXES_ORDER, LEVELS, RANDOM_STATE, load_scores, proxy_ground_truth,
    compute_compression_ratio, bootstrap_ci, safe_quantile_ranks,
    pairwise_accuracy,
)

# ── Output dirs ─────────────────────────────────────────────────
RES_DIR = ROOT / "output" / "mitigations" / "results"
FIG_DIR = ROOT / "output" / "mitigations" / "figures"
INT_DIR = ROOT / "output" / "mitigations" / "intermediate"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
INT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════
# QUANTILE NORMALISATION
# ═══════════════════════════════════════════════════════════════

def quantile_normalise_uniform(scores: np.ndarray) -> np.ndarray:
    """Map scores to Uniform[0, 10] via quantile ranks."""
    qr = safe_quantile_ranks(scores)
    return qr * 10.0


def quantile_normalise_beta(scores: np.ndarray, a: float = 2.0, b: float = 2.0) -> np.ndarray:
    """Map scores to Beta(a, b) scaled to [0, 10] via quantile ranks."""
    qr = safe_quantile_ranks(scores)
    qr = np.clip(qr, 1e-8, 1 - 1e-8)  # avoid ppf(0) / ppf(1) = ±inf
    return sp_stats.beta.ppf(qr, a, b) * 10.0


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def eval_mitigation(df_model: pd.DataFrame, score_col: str) -> dict:
    """Compute standard metrics for a single model against proxy ground truth."""
    scores = df_model[score_col].values
    target = proxy_ground_truth(df_model["level"].values)

    # Wasserstein distance to proxy GT
    wd = float(sp_stats.wasserstein_distance(scores, target))

    # Compression ratio
    cr = compute_compression_ratio(scores)

    # Spearman rho
    rho, rho_p = sp_stats.spearmanr(scores, target)

    # Dose-response slope: linear regression of score ~ level
    slope_res = sp_stats.linregress(df_model["level"].values, scores)
    slope = slope_res.slope

    # Pairwise accuracy
    pa = pairwise_accuracy(scores, target)

    return {
        "wasserstein": round(wd, 4),
        "compression_ratio": round(cr, 4),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rho_p),
        "dose_response_slope": round(float(slope), 4),
        "pairwise_accuracy": round(pa, 4),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading scores …")
    df = load_scores(ROOT)
    models = sorted(df["model"].unique())
    print(f"  {len(df)} rows, {len(models)} models")

    results_uniform, results_beta = [], []

    for model in models:
        sub = df[df["model"] == model].copy()
        raw = sub["score"].values.astype(float)

        # Normalise
        sub["score_uniform"] = quantile_normalise_uniform(raw)
        sub["score_beta"] = quantile_normalise_beta(raw)

        # Metrics: raw
        m_raw = eval_mitigation(sub, "score")
        m_raw.update({"model": model, "method": "raw"})

        # Metrics: uniform
        m_uni = eval_mitigation(sub, "score_uniform")
        m_uni.update({"model": model, "method": "quantile_uniform"})

        # Metrics: beta
        m_beta = eval_mitigation(sub, "score_beta")
        m_beta.update({"model": model, "method": "quantile_beta"})

        results_uniform.append(m_raw)
        results_uniform.append(m_uni)
        results_beta.append(m_raw)
        results_beta.append(m_beta)

        print(f"  {model:30s}  CR raw={m_raw['compression_ratio']:.3f}  "
              f"uniform={m_uni['compression_ratio']:.3f}  "
              f"beta={m_beta['compression_ratio']:.3f}")

    # Save results
    cols = ["model", "method", "wasserstein", "compression_ratio",
            "spearman_rho", "spearman_p", "dose_response_slope", "pairwise_accuracy"]
    pd.DataFrame(results_uniform)[cols].to_csv(
        RES_DIR / "quantile_uniform.csv", index=False)
    pd.DataFrame(results_beta)[cols].to_csv(
        RES_DIR / "quantile_beta.csv", index=False)
    print(f"\n  Saved → {RES_DIR / 'quantile_uniform.csv'}")
    print(f"  Saved → {RES_DIR / 'quantile_beta.csv'}")

    # ── Figure: KDE before/after ────────────────────────────────
    n_models = len(models)
    ncols = min(4, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["model"] == model]
        raw = sub["score"].values.astype(float)

        sns.kdeplot(raw, ax=ax, label="Raw", color="#d62728", linewidth=1.5)
        sns.kdeplot(quantile_normalise_uniform(raw), ax=ax,
                    label="Uniform", color="#2ca02c", linewidth=1.5)
        sns.kdeplot(quantile_normalise_beta(raw), ax=ax,
                    label="Beta(2,2)", color="#1f77b4", linewidth=1.5)

        ax.set_title(model, fontsize=10)
        ax.set_xlim(-0.5, 10.5)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Quantile Normalisation: Score Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quantile_kde.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'quantile_kde.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
