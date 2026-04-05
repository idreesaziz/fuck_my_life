"""
Step 5 — Analysis & Visualization Pipeline
Generates all figures and statistical summaries for the results section.

Outputs:
  1. Per-axis dose-response curves (degradation level vs LLM score), per model
  2. Cross-axis comparison (overlay all 4 axes per model)
  3. Batched vs isolated comparison
  4. Composite Q vs LLM score scatter (S-curve detection)
  5. Summary statistics table
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Data Assembly ────────────────────────────────────────────────

def build_dataframe(scored_samples: list[dict],
                    llm_results: list[dict]) -> pd.DataFrame:
    """Merge quality scores with LLM ratings into a single DataFrame."""

    # Build sample lookup
    sample_lookup = {}
    for s in scored_samples:
        sample_lookup[s["id"]] = {
            "source_title": s["source_title"],
            "axis": s["axis"],
            "level": s["level"],
            "repetition": s.get("repetition", 0),
            "grammar_score": s["grammar_score"],
            "coherence_score": s["coherence_score"],
            "information_score": s["information_score"],
            "lexical_score": s["lexical_score"],
            "Q": s["Q"],
        }

    rows = []
    for r in llm_results:
        sid = r["sample_id"]
        if sid not in sample_lookup:
            continue
        if r.get("score") is None:
            continue
        row = {
            **sample_lookup[sid],
            "model": r["model"],
            "condition": r["condition"],
            "llm_score": r["score"],
            "llm_repetition": r.get("repetition", 0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Figure 1: Per-Axis Dose-Response Curves ──────────────────────

def plot_dose_response(df: pd.DataFrame, output_dir: str):
    """One figure per axis: degradation level (x) vs mean LLM score (y),
    with separate lines per model. Shows expected vs actual."""
    axes = df["axis"].unique()
    models = sorted(df["model"].unique())
    colors = sns.color_palette("husl", len(models))

    for axis in axes:
        fig, ax = plt.subplots(figsize=(8, 5))
        axis_df = df[(df["axis"] == axis) & (df["condition"] == "isolated")]

        for i, model in enumerate(models):
            model_df = axis_df[axis_df["model"] == model]
            grouped = model_df.groupby("level")["llm_score"].agg(["mean", "sem"])
            grouped = grouped.reset_index()

            ax.errorbar(
                grouped["level"], grouped["mean"],
                yerr=grouped["sem"] * 1.96,
                label=model, color=colors[i],
                marker="o", capsize=3, linewidth=2,
            )

        # Plot objective Q for this axis
        q_col = f"{axis}_score"
        if q_col in axis_df.columns:
            q_grouped = axis_df.groupby("level")[q_col].mean().reset_index()
            # Scale Q (0-1) to band scale (1-9) for visual comparison
            q_scaled = q_grouped[q_col] * 8 + 1
            ax.plot(
                q_grouped["level"], q_scaled,
                "--k", linewidth=2, alpha=0.5, label="Objective Q (scaled)",
            )

        ax.set_xlabel("Degradation Level", fontsize=12)
        ax.set_ylabel("Mean LLM Score (1-9)", fontsize=12)
        ax.set_title(f"Dose-Response: {axis.capitalize()} Degradation", fontsize=14)
        ax.legend()
        ax.set_ylim(0.5, 9.5)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"dose_response_{axis}.png", dpi=150)
        plt.close(fig)

    print(f"  Saved dose-response plots to {output_dir}")


# ── Figure 2: Cross-Axis Comparison ──────────────────────────────

def plot_cross_axis(df: pd.DataFrame, output_dir: str):
    """One figure per model: overlay all 4 axes on same plot to compare
    model sensitivity across degradation types."""
    models = sorted(df["model"].unique())
    axes = sorted(df["axis"].unique())
    colors = sns.color_palette("Set2", len(axes))

    for model in models:
        fig, ax = plt.subplots(figsize=(8, 5))
        model_df = df[(df["model"] == model) & (df["condition"] == "isolated")]

        for i, axis in enumerate(axes):
            axis_df = model_df[model_df["axis"] == axis]
            grouped = axis_df.groupby("level")["llm_score"].agg(["mean", "sem"])
            grouped = grouped.reset_index()

            ax.errorbar(
                grouped["level"], grouped["mean"],
                yerr=grouped["sem"] * 1.96,
                label=axis.capitalize(), color=colors[i],
                marker="s", capsize=3, linewidth=2,
            )

        ax.set_xlabel("Degradation Level", fontsize=12)
        ax.set_ylabel("Mean LLM Score (1-9)", fontsize=12)
        ax.set_title(f"Cross-Axis Sensitivity: {model}", fontsize=14)
        ax.legend()
        ax.set_ylim(0.5, 9.5)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        safe_name = model.replace(" ", "_").lower()
        fig.savefig(Path(output_dir) / f"cross_axis_{safe_name}.png", dpi=150)
        plt.close(fig)

    print(f"  Saved cross-axis plots to {output_dir}")


# ── Figure 3: Batched vs Isolated ────────────────────────────────

def plot_batched_vs_isolated(df: pd.DataFrame, output_dir: str):
    """Compare score distributions between batched and isolated conditions."""
    conditions = df["condition"].unique()
    if len(conditions) < 2:
        print("  [SKIP] Only one condition present, skipping batched vs isolated plot")
        return

    models = sorted(df["model"].unique())

    fig, axes_arr = plt.subplots(1, len(models), figsize=(5 * len(models), 5),
                                  sharey=True)
    if len(models) == 1:
        axes_arr = [axes_arr]

    for i, model in enumerate(models):
        ax = axes_arr[i]
        model_df = df[df["model"] == model]

        for condition in ["isolated", "batched"]:
            cond_df = model_df[model_df["condition"] == condition]
            grouped = cond_df.groupby("level")["llm_score"].mean().reset_index()
            style = "-o" if condition == "isolated" else "--s"
            ax.plot(grouped["level"], grouped["mean"], style,
                    label=condition.capitalize(), linewidth=2)

        ax.set_xlabel("Degradation Level", fontsize=11)
        if i == 0:
            ax.set_ylabel("Mean LLM Score", fontsize=11)
        ax.set_title(model, fontsize=13)
        ax.legend()
        ax.set_ylim(0.5, 9.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Batched vs Isolated Scoring", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "batched_vs_isolated.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved batched vs isolated plot to {output_dir}")


# ── Figure 4: Q vs LLM Score (S-Curve Detection) ────────────────

def plot_q_vs_llm(df: pd.DataFrame, output_dir: str):
    """Scatter: objective Q (x) vs LLM score (y). The core S-curve hypothesis.
    Includes identity line (perfect calibration) for reference."""
    models = sorted(df["model"].unique())
    colors = sns.color_palette("husl", len(models))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, model in enumerate(models):
        model_df = df[(df["model"] == model) & (df["condition"] == "isolated")]
        # Scale Q to 1-9 band
        q_scaled = model_df["Q"] * 8 + 1
        ax.scatter(
            q_scaled, model_df["llm_score"],
            alpha=0.15, s=20, color=colors[i], label=model,
        )
        # Binned mean line
        bins = np.linspace(1, 9, 17)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        model_df = model_df.copy()
        model_df["q_bin"] = pd.cut(q_scaled, bins=bins, labels=bin_centers)
        binned = model_df.groupby("q_bin", observed=True)["llm_score"].mean()
        ax.plot(
            binned.index.astype(float), binned.values,
            "-o", color=colors[i], linewidth=2, markersize=5,
        )

    # Identity line
    ax.plot([1, 9], [1, 9], "--k", alpha=0.4, linewidth=1.5,
            label="Perfect calibration")

    ax.set_xlabel("Objective Q Score (scaled 1-9)", fontsize=12)
    ax.set_ylabel("LLM Score (1-9)", fontsize=12)
    ax.set_title("Score Compression: Objective Q vs LLM Rating", fontsize=14)
    ax.legend()
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "q_vs_llm_scurve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved Q vs LLM S-curve plot to {output_dir}")


# ── Summary Statistics ───────────────────────────────────────────

def compute_statistics(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Compute key statistics: compression ratio, MAE, correlation, per model."""
    records = []

    for model in sorted(df["model"].unique()):
        for condition in sorted(df["condition"].unique()):
            subset = df[(df["model"] == model) & (df["condition"] == condition)]
            if subset.empty:
                continue

            q_scaled = subset["Q"] * 8 + 1
            llm = subset["llm_score"]

            # Score compression ratio: std(llm) / std(q_scaled)
            compression = llm.std() / q_scaled.std() if q_scaled.std() > 0 else np.nan

            # Pearson correlation
            r, p_val = stats.pearsonr(q_scaled, llm)

            # MAE
            mae = mean_absolute_error(q_scaled, llm)

            # Mean bias (positive = LLM overrates)
            bias = (llm - q_scaled).mean()

            records.append({
                "model": model,
                "condition": condition,
                "n_samples": len(subset),
                "compression_ratio": round(compression, 3),
                "pearson_r": round(r, 3),
                "p_value": round(p_val, 6),
                "mae": round(mae, 3),
                "mean_bias": round(bias, 3),
                "llm_mean": round(llm.mean(), 2),
                "llm_std": round(llm.std(), 2),
                "q_mean": round(q_scaled.mean(), 2),
                "q_std": round(q_scaled.std(), 2),
            })

    stats_df = pd.DataFrame(records)
    stats_path = Path(output_dir) / "summary_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved summary statistics to {stats_path}")

    # Per-axis sensitivity
    axis_records = []
    for model in sorted(df["model"].unique()):
        for axis in sorted(df["axis"].unique()):
            subset = df[(df["model"] == model) & (df["axis"] == axis)
                        & (df["condition"] == "isolated")]
            if subset.empty:
                continue

            # Slope of LLM score vs degradation level (linear regression)
            if len(subset["level"].unique()) > 1:
                slope, intercept, r_val, p_val, se = stats.linregress(
                    subset["level"], subset["llm_score"]
                )
            else:
                slope = intercept = r_val = p_val = se = np.nan

            axis_records.append({
                "model": model,
                "axis": axis,
                "slope": round(slope, 3),
                "r_squared": round(r_val**2, 3),
                "p_value": round(p_val, 6),
                "score_at_0": round(intercept, 2),
                "score_at_0.8": round(intercept + slope * 0.8, 2),
                "score_range": round(abs(slope * 0.8), 2),
            })

    axis_df = pd.DataFrame(axis_records)
    axis_path = Path(output_dir) / "axis_sensitivity.csv"
    axis_df.to_csv(axis_path, index=False)
    print(f"  Saved axis sensitivity analysis to {axis_path}")

    return stats_df


# ── Orchestrator ─────────────────────────────────────────────────

def run(config: dict, scored_samples: list[dict], llm_results: list[dict]):
    """Generate all analysis outputs."""
    analysis_cfg = config["analysis"]
    output_dir = analysis_cfg["output_dir"]
    figures_dir = analysis_cfg["figures_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    print("[Analysis] Building dataframe...")
    df = build_dataframe(scored_samples, llm_results)

    if df.empty:
        print("[Analysis] No valid LLM scores to analyze. Skipping.")
        return

    # Save raw merged data
    df.to_csv(Path(output_dir) / "merged_data.csv", index=False)
    print(f"  Merged {len(df)} data points")

    print("[Analysis] Generating dose-response curves...")
    plot_dose_response(df, figures_dir)

    print("[Analysis] Generating cross-axis comparisons...")
    plot_cross_axis(df, figures_dir)

    print("[Analysis] Generating batched vs isolated comparison...")
    plot_batched_vs_isolated(df, figures_dir)

    print("[Analysis] Generating Q vs LLM S-curve plot...")
    plot_q_vs_llm(df, figures_dir)

    print("[Analysis] Computing summary statistics...")
    compute_statistics(df, output_dir)

    print("[Analysis] Done.")


# ═══════════════════════════════════════════════════════════════
# SHARED UTILITIES  (used by mitigation scripts)
# ═══════════════════════════════════════════════════════════════

RANDOM_STATE = 42
AXES_ORDER = ["grammar", "coherence", "information", "lexical"]
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
MAX_DEGRADATION = 0.8
IDEAL_RANGE = 10.0 * (1.0 - 0.0) - 10.0 * (1.0 - MAX_DEGRADATION)  # 8.0

MODEL_SIZE_B = {
    "phi4-mini": 3.8, "mistral-7b": 7.0, "qwen2.5-7b": 7.0,
    "llama3.1-8b": 8.0, "gemma2-9b": 9.0, "phi4-14b": 14.0,
    "qwen2.5-14b": 14.0, "gpt-oss-120b-fireworks": 116.8,
    "minimax-m2p1-fireworks": 228.7, "gpt-5-mini": 100.0,
    "gemini-3-flash": 100.0,
}
API_MODELS = {"gpt-5-mini", "gemini-3-flash"}


def load_scores(root: Path) -> pd.DataFrame:
    """Load all score JSONs + degraded_samples metadata into a single DataFrame.

    Columns returned: sample_id, model, condition, score, axis, level,
                      article, category, score_probs (dict or None).
    """
    samples = json.loads((root / "data/degraded/degraded_samples.json")
                         .read_text(encoding="utf-8"))
    meta = {}
    for s in samples:
        meta[s["id"]] = {
            "article": s["source_title"],
            "category": s.get("category", "unknown"),
            "axis": s["axis"],
            "level": s["level"],
        }

    scores_dir = root / "data" / "scores"
    rows = []
    for sf in sorted(scores_dir.glob("*.json")):
        records = json.loads(sf.read_text(encoding="utf-8"))
        if not records:
            continue
        for r in records:
            sid = r["sample_id"]
            if sid not in meta or r.get("score") is None:
                continue
            m = meta[sid]
            rows.append({
                "sample_id": sid,
                "model": r["model"],
                "condition": r.get("condition", "isolated"),
                "score": r["score"],
                "axis": m["axis"],
                "level": m["level"],
                "article": m["article"],
                "category": m["category"],
                "score_probs": r.get("score_probs"),
            })
    return pd.DataFrame(rows)


def proxy_ground_truth(level: float | np.ndarray) -> float | np.ndarray:
    """target_score = (1 − level / max_level) × 10."""
    return (1.0 - np.asarray(level) / MAX_DEGRADATION) * 10.0


def compute_compression_ratio(scores: np.ndarray) -> float:
    """(max − min) / 10 after clipping to [0, 10]."""
    s = np.clip(np.asarray(scores, dtype=float), 0, 10)
    return float((s.max() - s.min()) / 10.0)


def bootstrap_ci(x, stat_fn=np.mean, n_boot: int = 2000,
                 ci: float = 0.95, seed: int = RANDOM_STATE):
    """Return (point_estimate, ci_lo, ci_hi)."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    point = float(stat_fn(x))
    boots = np.array([stat_fn(rng.choice(x, len(x), replace=True))
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return point, float(lo), float(hi)


def safe_quantile_ranks(arr: np.ndarray) -> np.ndarray:
    """Rank-based quantile transform → [0, 1]. Ties get average rank."""
    from scipy.stats import rankdata
    ranks = rankdata(arr, method="average")
    return (ranks - 1) / max(len(arr) - 1, 1)


def pairwise_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    """Fraction of concordant pairs: sign(pred_i − pred_j) == sign(target_i − target_j)."""
    pred = np.asarray(predicted, dtype=float)
    tgt = np.asarray(target, dtype=float)
    n = len(pred)
    if n < 2:
        return float("nan")
    concordant = 0
    total = 0
    for i in range(n):
        dp = pred[i] - pred[i + 1:]
        dt = tgt[i] - tgt[i + 1:]
        mask = dt != 0
        concordant += np.sum(np.sign(dp[mask]) == np.sign(dt[mask]))
        total += mask.sum()
    return concordant / total if total > 0 else float("nan")
