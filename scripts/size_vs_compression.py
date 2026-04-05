#!/usr/bin/env python3
"""
Model Size vs Score Compression Analysis
=========================================
Analyzes the relationship between LLM parameter count and:
  1. Compression ratio  (mean_at_0.0 − mean_at_0.8) / ideal_range
  2. Average sensitivity slope (β₁ from score ~ level regression)
  3. Score‐range utilization (# of distinct score values used)
  4. Per-axis compression breakdown

Outputs:
  output/figures/size_compression_scatter.png
  output/figures/size_compression_multiaxis.png
  output/figures/size_compression_table.png
  output/analysis/size_vs_compression.json
  output/analysis/size_vs_compression.csv

Usage:  python scripts/size_vs_compression.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Paths ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "output" / "figures"
ANALYSIS_DIR = ROOT / "output" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Known model sizes (billions of parameters) ─────────────────
# Sources: config_laptop_mine.yaml, config_laptop_friend.yaml,
#          and public model cards for API models.
MODEL_SIZE_B = {
    "phi4-mini":                3.8,
    "mistral-7b":               7.0,
    "qwen2.5-7b":               7.0,
    "llama3.1-8b":              8.0,
    "gemma2-9b":                9.0,
    "phi4-14b":                14.0,
    "qwen2.5-14b":             14.0,
    "gpt-oss-120b-fireworks": 116.8,
    "minimax-m2p1-fireworks": 228.7,
    # API models — sizes are public estimates; flagged separately
    "gpt-5-mini":             100.0,   # estimated
    "gemini-3-flash":         100.0,   # estimated
}
API_MODELS = {"gpt-5-mini", "gemini-3-flash"}

AXES_ORDER = ["grammar", "coherence", "information", "lexical"]
AXIS_LABELS = {"grammar": "Grammar", "coherence": "Coherence",
               "information": "Information", "lexical": "Lexical"}
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
IDEAL_RANGE = 8.0  # 10×(1−0) − 10×(1−0.8) = 10 − 2 = 8

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10,
})


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_scores():
    """Load all score JSONs and build a unified DataFrame."""
    samples = json.load(open(ROOT / "data/degraded/degraded_samples.json",
                             encoding="utf-8"))
    meta = {}
    for s in samples:
        meta[s["id"]] = {
            "article": s["source_title"],
            "axis": s["axis"],
            "level": s["level"],
        }

    scores_dir = ROOT / "data" / "scores"
    rows = []
    for sf in sorted(scores_dir.glob("*.json")):
        records = json.load(open(sf, encoding="utf-8"))
        if not records:
            continue
        model_name = records[0]["model"]
        for r in records:
            sid = r["sample_id"]
            if sid not in meta or r.get("score") is None:
                continue
            rows.append({**meta[sid], "sample_id": sid,
                         "model": model_name, "score": r["score"]})
    return pd.DataFrame(rows)


def compute_metrics(df):
    """Compute compression ratio, avg slope, range util per model."""
    results = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        size = MODEL_SIZE_B.get(model)
        if size is None:
            continue
        is_api = model in API_MODELS

        # Overall compression ratio
        mean_0 = sub[sub["level"] == 0.0]["score"].mean()
        mean_08 = sub[sub["level"] == 0.8]["score"].mean()
        compression = (mean_0 - mean_08) / IDEAL_RANGE

        # Average sensitivity slope across axes
        slopes = {}
        intercepts = {}
        for axis in AXES_ORDER:
            ax_sub = sub[sub["axis"] == axis]
            if len(ax_sub) < 5:
                continue
            r = sp_stats.linregress(ax_sub["level"], ax_sub["score"])
            slopes[axis] = r.slope
            intercepts[axis] = r.intercept
        avg_slope = float(np.mean(list(slopes.values()))) if slopes else float("nan")

        # Score-range utilization
        unique_scores = sorted(sub["score"].unique())
        range_used = len(unique_scores)
        score_std = float(sub["score"].std())

        # Per-axis compression
        axis_compression = {}
        for axis in AXES_ORDER:
            ax_sub = sub[sub["axis"] == axis]
            m0 = ax_sub[ax_sub["level"] == 0.0]["score"].mean()
            m8 = ax_sub[ax_sub["level"] == 0.8]["score"].mean()
            axis_compression[axis] = (m0 - m8) / IDEAL_RANGE

        results.append({
            "model": model,
            "size_b": size,
            "is_api": is_api,
            "n_scores": len(sub),
            "mean_at_0": float(mean_0),
            "mean_at_08": float(mean_08),
            "compression_ratio": float(compression),
            "avg_slope": avg_slope,
            "slopes": slopes,
            "range_used": range_used,
            "score_std": score_std,
            **{f"compression_{a}": axis_compression[a] for a in AXES_ORDER},
        })

    return sorted(results, key=lambda x: x["size_b"])


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Scatter — compression ratio & slope vs size
# ═══════════════════════════════════════════════════════════════

def fig_scatter(metrics):
    """Two-panel scatter: compression ratio and avg slope vs model size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    local = [m for m in metrics if not m["is_api"]]
    api = [m for m in metrics if m["is_api"]]

    for ax, ykey, ylabel, title, ref_val, ref_label in [
        (axes[0], "compression_ratio", "Compression Ratio",
         "Score Compression vs Model Size", 1.0, "Perfect (1.0)"),
        (axes[1], "avg_slope", "Avg Sensitivity Slope (β₁)",
         "Sensitivity Slope vs Model Size", -10.0, "Perfect (−10)"),
    ]:
        # Local models
        xs_local = [m["size_b"] for m in local]
        ys_local = [m[ykey] for m in local]
        ax.scatter(xs_local, ys_local, s=120, c="#1f77b4", edgecolors="white",
                   zorder=3, label="Local/open-weight")
        for m in local:
            ax.annotate(m["model"], (m["size_b"], m[ykey]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="#1f77b4")

        # API models
        if api:
            xs_api = [m["size_b"] for m in api]
            ys_api = [m[ykey] for m in api]
            ax.scatter(xs_api, ys_api, s=120, c="#e6550d", marker="D",
                       edgecolors="white", zorder=3, label="API (est. size)")
            for m in api:
                ax.annotate(m["model"], (m["size_b"], m[ykey]),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=8, color="#e6550d")

        # Trend line (local only, reliable sizes)
        if len(xs_local) >= 3:
            log_x = np.log10(xs_local)
            slope, intercept, r, p, se = sp_stats.linregress(log_x, ys_local)
            xs_fit = np.linspace(min(xs_local) * 0.8, max(xs_local) * 1.2, 200)
            ys_fit = intercept + slope * np.log10(xs_fit)
            ax.plot(xs_fit, ys_fit, "--", color="#1f77b4", alpha=0.5, lw=1.5,
                    label=f"log-linear fit (r={r:.2f}, p={p:.3f})")

        # All-model trend (if API included)
        if len(metrics) >= 4:
            all_x = np.log10([m["size_b"] for m in metrics])
            all_y = [m[ykey] for m in metrics]
            sl, ic, r_all, p_all, _ = sp_stats.linregress(all_x, all_y)
            xs_all = np.linspace(min(m["size_b"] for m in metrics) * 0.8,
                                 max(m["size_b"] for m in metrics) * 1.2, 200)
            ax.plot(xs_all, ic + sl * np.log10(xs_all), ":",
                    color="gray", alpha=0.5, lw=1.3,
                    label=f"all models (r={r_all:.2f}, p={p_all:.3f})")

        ax.axhline(ref_val, ls="-.", color="green", alpha=0.4, lw=1,
                   label=ref_label)
        ax.set_xscale("log")
        ax.set_xlabel("Model Size (B parameters, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.suptitle("Model Scale ↔ Score Compression Relationship",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "size_compression_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Per-axis compression vs size
# ═══════════════════════════════════════════════════════════════

def fig_multiaxis(metrics):
    """Per-axis compression ratio vs model size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    axis_colors = {"grammar": "#1f77b4", "coherence": "#ff7f0e",
                   "information": "#2ca02c", "lexical": "#d62728"}
    markers = {"grammar": "o", "coherence": "s",
               "information": "^", "lexical": "D"}

    for axis in AXES_ORDER:
        xs = [m["size_b"] for m in metrics]
        ys = [m[f"compression_{axis}"] for m in metrics]
        ax.scatter(xs, ys, s=90, color=axis_colors[axis],
                   marker=markers[axis], label=AXIS_LABELS[axis],
                   edgecolors="white", zorder=3)

        # Log-linear fit
        if len(xs) >= 3:
            log_x = np.log10(xs)
            sl, ic, r, p, _ = sp_stats.linregress(log_x, ys)
            xf = np.linspace(min(xs) * 0.8, max(xs) * 1.2, 200)
            ax.plot(xf, ic + sl * np.log10(xf), "--",
                    color=axis_colors[axis], alpha=0.4, lw=1.3)

    ax.axhline(1.0, ls="-.", color="green", alpha=0.4, lw=1,
               label="Perfect calibration (1.0)")
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B parameters, log scale)")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Per-Axis Score Compression vs Model Size",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    fig.tight_layout()
    path = FIG_DIR / "size_compression_multiaxis.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Summary table as figure
# ═══════════════════════════════════════════════════════════════

def fig_table(metrics):
    """Render a summary table as a publication-quality figure."""
    cols = ["Model", "Size (B)", "Type", "Comp.\nRatio", "Avg\nSlope",
            "Gram.", "Coh.", "Info.", "Lex.", "σ(score)"]
    rows = []
    for m in metrics:
        rows.append([
            m["model"],
            f"{m['size_b']:.1f}",
            "API" if m["is_api"] else "Open",
            f"{m['compression_ratio']:.3f}",
            f"{m['avg_slope']:.2f}",
            f"{m['compression_grammar']:.3f}",
            f"{m['compression_coherence']:.3f}",
            f"{m['compression_information']:.3f}",
            f"{m['compression_lexical']:.3f}",
            f"{m['score_std']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(16, 0.5 + 0.45 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Style header
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor("#2171b5")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(color)

    fig.suptitle("Model Size vs Score Compression — Summary",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = FIG_DIR / "size_compression_table.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_statistics(metrics):
    """Compute correlations and print summary."""
    local = [m for m in metrics if not m["is_api"]]
    all_m = metrics

    results = {}

    for label, subset in [("local_only", local), ("all_models", all_m)]:
        if len(subset) < 3:
            continue
        log_sizes = np.log10([m["size_b"] for m in subset])
        comp = [m["compression_ratio"] for m in subset]
        slopes = [m["avg_slope"] for m in subset]

        # Pearson on log(size) vs compression
        r_comp, p_comp = sp_stats.pearsonr(log_sizes, comp)
        # Spearman on raw size vs compression
        rho_comp, rho_p_comp = sp_stats.spearmanr(
            [m["size_b"] for m in subset], comp)
        # Pearson on log(size) vs slope
        r_slope, p_slope = sp_stats.pearsonr(log_sizes, slopes)
        # Kendall's tau
        tau_comp, tau_p = sp_stats.kendalltau(
            [m["size_b"] for m in subset], comp)

        info = {
            "n": len(subset),
            "pearson_logsize_vs_compression": {"r": float(r_comp), "p": float(p_comp)},
            "spearman_size_vs_compression": {"rho": float(rho_comp), "p": float(rho_p_comp)},
            "kendall_size_vs_compression": {"tau": float(tau_comp), "p": float(tau_p)},
            "pearson_logsize_vs_slope": {"r": float(r_slope), "p": float(p_slope)},
        }
        results[label] = info

        print(f"\n  ── {label} (n={len(subset)}) ──")
        print(f"  Pearson  log₁₀(size) vs compression:  r={r_comp:+.3f}  p={p_comp:.4f}")
        print(f"  Spearman size vs compression:          ρ={rho_comp:+.3f}  p={rho_p_comp:.4f}")
        print(f"  Kendall  size vs compression:          τ={tau_comp:+.3f}  p={tau_p:.4f}")
        print(f"  Pearson  log₁₀(size) vs avg slope:    r={r_slope:+.3f}  p={p_slope:.4f}")

    # Per-axis correlations (all models)
    if len(all_m) >= 3:
        print("\n  ── Per-axis Spearman (all models) ──")
        axis_corr = {}
        for axis in AXES_ORDER:
            sizes = [m["size_b"] for m in all_m]
            comps = [m[f"compression_{axis}"] for m in all_m]
            rho, p = sp_stats.spearmanr(sizes, comps)
            axis_corr[axis] = {"rho": float(rho), "p": float(p)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {AXIS_LABELS[axis]:12s}  ρ={rho:+.3f}  p={p:.4f}  {sig}")
        results["per_axis_spearman"] = axis_corr

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print(" Model Size vs Score Compression Analysis")
    print("=" * 64)

    print("\nLoading data...")
    df = load_scores()
    print(f"  {len(df)} scored samples across {df['model'].nunique()} models")

    print("\nComputing metrics...")
    metrics = compute_metrics(df)
    for m in metrics:
        print(f"  {m['model']:30s}  {m['size_b']:7.1f}B  "
              f"comp={m['compression_ratio']:.3f}  "
              f"slope={m['avg_slope']:+.2f}  "
              f"σ={m['score_std']:.2f}")

    print("\nStatistical tests...")
    stats = run_statistics(metrics)

    print("\nGenerating figures...")
    fig_scatter(metrics)
    fig_multiaxis(metrics)
    fig_table(metrics)

    # Save JSON
    output = {"metrics": metrics, "statistics": stats}
    json_path = ANALYSIS_DIR / "size_vs_compression.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  → Saved {json_path.relative_to(ROOT)}")

    # Save CSV
    csv_rows = []
    for m in metrics:
        csv_rows.append({
            "model": m["model"],
            "size_b": m["size_b"],
            "is_api": m["is_api"],
            "compression_ratio": m["compression_ratio"],
            "avg_slope": m["avg_slope"],
            "compression_grammar": m["compression_grammar"],
            "compression_coherence": m["compression_coherence"],
            "compression_information": m["compression_information"],
            "compression_lexical": m["compression_lexical"],
            "score_std": m["score_std"],
            "range_used": m["range_used"],
            "n_scores": m["n_scores"],
        })
    csv_path = ANALYSIS_DIR / "size_vs_compression.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  → Saved {csv_path.relative_to(ROOT)}")

    print(f"\n{'=' * 64}")
    print(" DONE")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
