"""
Generate all analysis graphs from GPT-5 mini scoring data.
Works directly from degraded_samples.json + gpt5_mini_scores.json.

Usage: python scripts/generate_graphs.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Load & merge data ────────────────────────────────────────────

samples = json.load(open("data/degraded/degraded_samples.json", "r", encoding="utf-8"))
scores = json.load(open("data/scores/gpt5_mini_scores.json", "r", encoding="utf-8"))

score_map = {s["sample_id"]: s["score"] for s in scores}

rows = []
for s in samples:
    sid = s["id"]
    if sid not in score_map or score_map[sid] is None:
        continue
    rows.append({
        "id": sid,
        "source_title": s["source_title"],
        "axis": s["axis"],
        "level": s["level"],
        "repetition": s["repetition"],
        "llm_score": score_map[sid],
    })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} data points")
print(f"Axes: {sorted(df['axis'].unique())}")
print(f"Levels: {sorted(df['level'].unique())}")
print(f"Articles: {df['source_title'].nunique()}")

out = Path("output/figures")
out.mkdir(parents=True, exist_ok=True)

AXIS_LABELS = {
    "grammar": "Grammar",
    "coherence": "Coherence",
    "information": "Information",
    "lexical": "Lexical Diversity",
}
LEVELS = sorted(df["level"].unique())

# ── Style ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2", 4)

# =====================================================================
# Figure 1: Per-Axis Dose-Response Curves (4 subplots)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
axes = axes.flatten()

for i, axis in enumerate(["grammar", "coherence", "information", "lexical"]):
    ax = axes[i]
    axis_df = df[df["axis"] == axis]
    grouped = axis_df.groupby("level")["llm_score"].agg(["mean", "sem", "std"]).reset_index()

    ax.errorbar(
        grouped["level"], grouped["mean"],
        yerr=grouped["sem"] * 1.96,
        marker="o", capsize=4, linewidth=2.5, color=PALETTE[i],
        markersize=8, label="GPT-5 mini",
    )
    # shade CI band
    ax.fill_between(
        grouped["level"],
        grouped["mean"] - grouped["sem"] * 1.96,
        grouped["mean"] + grouped["sem"] * 1.96,
        alpha=0.15, color=PALETTE[i],
    )
    ax.set_title(AXIS_LABELS[axis], fontsize=14, fontweight="bold")
    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("Mean LLM Score (0–10)")
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(LEVELS)
    ax.axhline(y=5, color="gray", linestyle=":", alpha=0.5, label="Midpoint (5)")

fig.suptitle("Dose-Response: Degradation Level vs LLM Score", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out / "dose_response_per_axis.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  [1/8] dose_response_per_axis.png")


# =====================================================================
# Figure 2: Cross-Axis Comparison (all 4 axes overlaid)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 6))

for i, axis in enumerate(["grammar", "coherence", "information", "lexical"]):
    axis_df = df[df["axis"] == axis]
    grouped = axis_df.groupby("level")["llm_score"].agg(["mean", "sem"]).reset_index()
    ax.errorbar(
        grouped["level"], grouped["mean"],
        yerr=grouped["sem"] * 1.96,
        marker="s", capsize=4, linewidth=2.5, color=PALETTE[i],
        markersize=7, label=AXIS_LABELS[axis],
    )

ax.set_xlabel("Degradation Level", fontsize=13)
ax.set_ylabel("Mean LLM Score (0–10)", fontsize=13)
ax.set_title("Cross-Axis Sensitivity: GPT-5 mini", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(-0.5, 10.5)
ax.set_xticks(LEVELS)
ax.axhline(y=5, color="gray", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(out / "cross_axis_comparison.png", dpi=200)
plt.close(fig)
print("  [2/8] cross_axis_comparison.png")


# =====================================================================
# Figure 3: Score Distribution Histogram (all data)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
score_counts = df["llm_score"].value_counts().sort_index()
bars = ax.bar(score_counts.index, score_counts.values, color=sns.color_palette("Blues_d", len(score_counts)),
              edgecolor="white", linewidth=0.8)

# Annotate counts on bars
for bar, count in zip(bars, score_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            str(count), ha="center", va="bottom", fontsize=9)

ax.set_xlabel("LLM Score", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Score Distribution: GPT-5 mini (n=9,000)", fontsize=15, fontweight="bold")
ax.set_xticks(range(0, 11))
fig.tight_layout()
fig.savefig(out / "score_distribution.png", dpi=200)
plt.close(fig)
print("  [3/8] score_distribution.png")


# =====================================================================
# Figure 4: Box Plots by Degradation Level (per axis)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

for i, axis in enumerate(["grammar", "coherence", "information", "lexical"]):
    ax = axes[i]
    axis_df = df[df["axis"] == axis]
    sns.boxplot(data=axis_df, x="level", y="llm_score", ax=ax, palette="Set2",
                fliersize=2, width=0.6)
    ax.set_title(AXIS_LABELS[axis], fontsize=14, fontweight="bold")
    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("LLM Score (0–10)")
    ax.set_ylim(-0.5, 10.5)

fig.suptitle("Score Distributions by Degradation Level", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out / "boxplots_per_axis.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  [4/8] boxplots_per_axis.png")


# =====================================================================
# Figure 5: Violin Plots — Score Compression Visualization
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x="level", y="llm_score", hue="axis",
               palette="Set2", inner="box", ax=ax, density_norm="width")
ax.set_xlabel("Degradation Level", fontsize=13)
ax.set_ylabel("LLM Score (0–10)", fontsize=13)
ax.set_title("Score Compression Across Axes", fontsize=15, fontweight="bold")
ax.legend(title="Axis", labels=[AXIS_LABELS[a] for a in sorted(df["axis"].unique())])
ax.set_ylim(-0.5, 10.5)
fig.tight_layout()
fig.savefig(out / "violin_compression.png", dpi=200)
plt.close(fig)
print("  [5/8] violin_compression.png")


# =====================================================================
# Figure 6: Heatmap — Mean Score by (Axis, Level)
# =====================================================================
pivot = df.pivot_table(values="llm_score", index="axis", columns="level", aggfunc="mean")
pivot.index = [AXIS_LABELS.get(a, a) for a in pivot.index]

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=10,
            linewidths=1, ax=ax, cbar_kws={"label": "Mean LLM Score"})
ax.set_xlabel("Degradation Level", fontsize=13)
ax.set_ylabel("Axis", fontsize=13)
ax.set_title("Mean LLM Score Heatmap", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(out / "heatmap_scores.png", dpi=200)
plt.close(fig)
print("  [6/8] heatmap_scores.png")


# =====================================================================
# Figure 7: Undegraded Score Distribution (level=0 only)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
undeg = df[df["level"] == 0.0]
score_counts_0 = undeg["llm_score"].value_counts().sort_index()
bars = ax.bar(score_counts_0.index, score_counts_0.values,
              color=sns.color_palette("Greens_d", len(score_counts_0)),
              edgecolor="white", linewidth=0.8)
for bar, count in zip(bars, score_counts_0.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(count), ha="center", va="bottom", fontsize=9)

ax.set_xlabel("LLM Score", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title(f"Undegraded Text Scores (level=0, n={len(undeg)})\nMean={undeg['llm_score'].mean():.2f}, "
             f"Never scores 0 or 10", fontsize=14, fontweight="bold")
ax.set_xticks(range(0, 11))
ax.axvline(x=undeg["llm_score"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean={undeg['llm_score'].mean():.1f}")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(out / "undegraded_distribution.png", dpi=200)
plt.close(fig)
print("  [7/8] undegraded_distribution.png")


# =====================================================================
# Figure 8: Per-Axis Slope Analysis (how much does score drop per axis)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

slopes = []
for axis in ["grammar", "coherence", "information", "lexical"]:
    axis_df = df[df["axis"] == axis]
    slope, intercept, r_val, p_val, se = stats.linregress(axis_df["level"], axis_df["llm_score"])
    slopes.append({
        "axis": AXIS_LABELS[axis],
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_val ** 2,
        "p_value": p_val,
    })

slopes_df = pd.DataFrame(slopes).sort_values("slope")

bars = ax.barh(slopes_df["axis"], slopes_df["slope"], color=PALETTE[:4], edgecolor="black", linewidth=0.8)
for bar, row in zip(bars, slopes_df.itertuples()):
    ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
            f"R²={row.r_squared:.2f}", va="center", fontsize=10, fontweight="bold", color="white")

ax.set_xlabel("Slope (score change per unit degradation)", fontsize=12)
ax.set_title("Axis Sensitivity: How Much Does Score Drop?", fontsize=14, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.8)
fig.tight_layout()
fig.savefig(out / "axis_sensitivity_slopes.png", dpi=200)
plt.close(fig)
print("  [8/8] axis_sensitivity_slopes.png")


# =====================================================================
# Figure 9: Score Distribution Per Axis (histograms)
# =====================================================================
fig, axes_arr = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes_arr = axes_arr.flatten()

for i, axis in enumerate(["grammar", "coherence", "information", "lexical"]):
    ax = axes_arr[i]
    axis_df = df[df["axis"] == axis]
    counts = axis_df["llm_score"].value_counts().sort_index().reindex(range(0, 11), fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=PALETTE[i], edgecolor="white", linewidth=0.8, alpha=0.85)
    for bar, count in zip(bars, counts.values):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{AXIS_LABELS[axis]} (n={len(axis_df)}, mean={axis_df['llm_score'].mean():.1f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("LLM Score")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, 11))
    ax.axvline(x=axis_df["llm_score"].mean(), color="red", linestyle="--", linewidth=1.5, alpha=0.7)

fig.suptitle("Score Distribution Per Axis: GPT-5 mini", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out / "distribution_per_axis.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  [9/9] distribution_per_axis.png")


# =====================================================================
# Summary Statistics to CSV
# =====================================================================
print("\n=== SUMMARY STATISTICS ===")
summary_rows = []
for axis in ["grammar", "coherence", "information", "lexical"]:
    axis_df = df[df["axis"] == axis]
    slope, intercept, r_val, p_val, se = stats.linregress(axis_df["level"], axis_df["llm_score"])
    for level in LEVELS:
        lvl_df = axis_df[axis_df["level"] == level]
        summary_rows.append({
            "axis": axis,
            "level": level,
            "mean": round(lvl_df["llm_score"].mean(), 2),
            "std": round(lvl_df["llm_score"].std(), 2),
            "median": lvl_df["llm_score"].median(),
            "min": lvl_df["llm_score"].min(),
            "max": lvl_df["llm_score"].max(),
            "n": len(lvl_df),
        })
    print(f"  {axis:12s} | slope={slope:+.2f} | R²={r_val**2:.3f} | "
          f"level0={axis_df[axis_df['level']==0.0]['llm_score'].mean():.1f} → "
          f"level0.8={axis_df[axis_df['level']==0.8]['llm_score'].mean():.1f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(out / "summary_statistics.csv", index=False)

# Overall compression stat
all_0 = df[df["level"] == 0.0]["llm_score"]
all_08 = df[df["level"] == 0.8]["llm_score"]
print(f"\n  Overall: undegraded mean={all_0.mean():.2f}, std={all_0.std():.2f}")
print(f"  Overall: max degraded mean={all_08.mean():.2f}, std={all_08.std():.2f}")
print(f"  Score range used: {df['llm_score'].min()} to {df['llm_score'].max()} "
      f"(of 0-10 scale)")
print(f"  Never scored: {set(range(11)) - set(df['llm_score'].unique())}")

print(f"\nAll figures saved to {out}/")
