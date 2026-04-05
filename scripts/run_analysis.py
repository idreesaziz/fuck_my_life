#!/usr/bin/env python3
"""
No One Gets an A — Comprehensive Statistical Analysis
====================================================
19 tests + 14 graphs (publication-quality)

Primary outcomes:
  1. Compression ratio (mean_at_0.0 - mean_at_0.8) / ideal_range per model
  2. Average linear slope across axes per model
  3. Median paired difference (Gemini - GPT)

Usage:  python scripts/run_analysis.py

Output:
  output/figures/*.png         — 14 publication-quality figures
  output/analysis/results.json — structured test results
  output/analysis/*.csv        — summary tables
"""

import json
import math
import sys
import warnings
import yaml
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "output" / "figures"
ANALYSIS_DIR = ROOT / "output" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.figsize": (10, 6),
})
_FIXED_COLORS = {"gpt-5-mini": "#2171b5", "gemini-3-flash": "#e6550d"}
_COLOR_PALETTE = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
COLORS = {}       # populated in main() after data load
AXES_ORDER = ["grammar", "coherence", "information", "lexical"]
AXIS_LABELS = {"grammar": "Grammar", "coherence": "Coherence",
               "information": "Information", "lexical": "Lexical"}
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
MODEL_NAMES = []  # populated in main() after data load
MODEL_SIZE_B = {}  # model_name -> size in billions (local models only)

RESULTS = {}  # collect all test results


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Build unified + paired DataFrames from all score JSON files under data/scores/."""
    samples = json.load(open(ROOT / "data/degraded/degraded_samples.json",
                             encoding="utf-8"))

    meta = {}
    for s in samples:
        meta[s["id"]] = {
            "article": s["source_title"],
            "category": s.get("category", "unknown"),
            "axis": s["axis"],
            "level": s["level"],
            "rep": s["repetition"],
            "article_length": len(s["original_text"]),
        }

    # Discover all score files and group by model name (inferred from records)
    scores_dir = ROOT / "data" / "scores"
    all_scores = {}  # model_name -> list[dict]
    for sf in sorted(scores_dir.glob("*.json")):
        records = json.load(open(sf, encoding="utf-8"))
        if not records:
            continue
        model_name = records[0]["model"]
        all_scores[model_name] = records
        print(f"  [load_data] {sf.name}: {len(records)} records ({model_name})")

    rows = []
    for model_name, records in all_scores.items():
        for r in records:
            sid = r["sample_id"]
            if sid not in meta or r.get("score") is None:
                continue
            rows.append({**meta[sid], "sample_id": sid,
                         "model": model_name, "score": r["score"]})

    df = pd.DataFrame(rows)

    # Paired DataFrame: GPT vs Gemini for backward-compatible inter-model tests
    gpt_name = next((n for n in all_scores if "gpt" in n.lower()), None)
    gem_name = next((n for n in all_scores if "gemini" in n.lower()), None)

    paired = pd.DataFrame()
    if gpt_name and gem_name:
        gpt_map = {r["sample_id"]: r["score"] for r in all_scores[gpt_name]}
        gem_map = {r["sample_id"]: r["score"] for r in all_scores[gem_name]}
        paired_rows = []
        for sid, m in meta.items():
            if (sid in gpt_map and sid in gem_map
                    and gpt_map[sid] is not None and gem_map[sid] is not None):
                paired_rows.append({
                    **m, "sample_id": sid,
                    "gpt_score": gpt_map[sid], "gemini_score": gem_map[sid],
                    "diff": gem_map[sid] - gpt_map[sid],
                })
        paired = pd.DataFrame(paired_rows)

    # ── Build logprob DataFrame (local models only) ───────────
    lp_rows = []
    for model_name, records in all_scores.items():
        for r in records:
            sp = r.get("score_probs")
            if not sp:
                continue
            sid = r["sample_id"]
            if sid not in meta:
                continue
            m = meta[sid]
            sp_int = {int(k): float(v) for k, v in sp.items()}
            h = 0.0
            for p in sp_int.values():
                if p > 0:
                    h -= p * math.log2(p)
            lp_rows.append({
                "sample_id": sid,
                "model": model_name,
                "axis": m["axis"],
                "level": m["level"],
                "entropy": h,
                "p_boundary": sp_int.get(0, 0.0) + sp_int.get(10, 0.0),
                "p_0": sp_int.get(0, 0.0),
                "p_10": sp_int.get(10, 0.0),
                **{f"p_{i}": sp_int.get(i, 0.0) for i in range(11)},
            })
    lp_df = pd.DataFrame(lp_rows) if lp_rows else pd.DataFrame()

    return df, paired, lp_df


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def header(title):
    print(f"\n{'═' * 64}\n {title}\n{'═' * 64}")


def cliffs_delta(x, y):
    """Non-parametric effect size for two independent groups."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    total = 0
    chunk = 5000
    for i in range(0, len(x), chunk):
        diff = x[i:i + chunk, None] - y[None, :]
        total += np.sum(np.sign(diff))
    return total / (len(x) * len(y))


def cohens_d(x, y):
    """Cohen's d for two independent groups."""
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                      (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0


def rank_biserial(x, y):
    """Matched-pairs rank-biserial r from Wilcoxon signed-rank test."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0.0
    stat, p = sp_stats.wilcoxon(d)
    # r = 1 - 2W / (n(n+1)/2)
    r = 1.0 - (2.0 * stat) / (n * (n + 1) / 2.0)
    return stat, p, r


def compute_icc_oneway(groups):
    """ICC(1,1) from one-way random effects ANOVA."""
    k = len(groups[0])  # reps per group
    n = len(groups)
    grand = np.mean([np.mean(g) for g in groups])
    ms_between = k * sum((np.mean(g) - grand) ** 2 for g in groups) / (n - 1)
    ms_within = sum(sum((x - np.mean(g)) ** 2 for x in g)
                    for g in groups) / (n * (k - 1))
    if (ms_between + (k - 1) * ms_within) == 0:
        return 0.0
    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)


def block_bootstrap_means(df_sub, n_boot=10000, seed=42):
    """Block-bootstrap mean by resampling articles."""
    art_means = df_sub.groupby("article")["score"].mean()
    vals = art_means.values
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    n = len(vals)
    for i in range(n_boot):
        boots[i] = vals[rng.randint(0, n, size=n)].mean()
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════
# T1: DISTRIBUTION SUMMARY
# ═══════════════════════════════════════════════════════════════

def test_t1(df):
    header("T1: Distribution Summary")
    res = {}
    for model in MODEL_NAMES:
        s = df[df["model"] == model]["score"].values
        c = Counter(s.tolist())
        info = {
            "n": len(s), "mean": float(np.mean(s)), "std": float(np.std(s, ddof=1)),
            "min": int(np.min(s)), "max": int(np.max(s)),
            "skewness": float(sp_stats.skew(s)),
            "kurtosis": float(sp_stats.kurtosis(s)),
            "distribution": {k: c.get(k, 0) for k in range(11)},
        }
        res[model] = info
        print(f"\n  {model}:")
        print(f"    n={info['n']}  mean={info['mean']:.2f}  std={info['std']:.2f}")
        print(f"    range={info['min']}-{info['max']}  skew={info['skewness']:.3f}  "
              f"kurtosis={info['kurtosis']:.3f}")
        print(f"    dist: {info['distribution']}")
    RESULTS["T1"] = res


# ═══════════════════════════════════════════════════════════════
# T2: BOUNDARY AVOIDANCE (calibrated baseline)
# ═══════════════════════════════════════════════════════════════

def test_t2(df):
    header("T2: Boundary Avoidance (calibrated baseline)")
    res = {}
    for model in MODEL_NAMES:
        s = df[df["model"] == model]["score"].values
        n0, n10 = int(np.sum(s == 0)), int(np.sum(s == 10))
        n1, n9 = int(np.sum(s == 1)), int(np.sum(s == 9))

        # Calibrated baseline: fit linear model, simulate from it
        sub = df[df["model"] == model]
        slope_fit = sp_stats.linregress(sub["level"], sub["score"])
        residual_std = np.std(sub["score"] - (slope_fit.intercept +
                                               slope_fit.slope * sub["level"]))
        rng = np.random.RandomState(42)
        n_sim = 100
        sim_zeros, sim_tens = [], []
        for _ in range(n_sim):
            predicted = slope_fit.intercept + slope_fit.slope * sub["level"].values
            sim_scores = np.clip(np.round(predicted +
                                          rng.normal(0, residual_std, len(predicted))),
                                 0, 10).astype(int)
            sim_zeros.append(np.sum(sim_scores == 0))
            sim_tens.append(np.sum(sim_scores == 10))

        info = {
            "count_0": n0, "count_1": n1, "count_9": n9, "count_10": n10,
            "simulated_mean_0s": float(np.mean(sim_zeros)),
            "simulated_mean_10s": float(np.mean(sim_tens)),
        }
        res[model] = info
        print(f"\n  {model}:")
        print(f"    Observed:  0s={n0}, 1s={n1}, 9s={n9}, 10s={n10}")
        print(f"    Simulated (calibrated linear + noise, n=100 sims):")
        print(f"      Expected 0s: {np.mean(sim_zeros):.1f}  Expected 10s: {np.mean(sim_tens):.1f}")
        if n0 == 0 and np.mean(sim_zeros) > 0:
            print(f"    ⇒ Model avoids score 0 (expected ~{np.mean(sim_zeros):.0f}, got 0)")
        if n10 == 0 and np.mean(sim_tens) > 0:
            print(f"    ⇒ Model avoids score 10 (expected ~{np.mean(sim_tens):.0f}, got 0)")
    RESULTS["T2"] = res


# ═══════════════════════════════════════════════════════════════
# T3: COMPRESSION RATIO
# ═══════════════════════════════════════════════════════════════

def test_t3(df):
    header("T3: Compression Ratio")
    # Perfect calibration: level 0.0 → 10, level 0.8 → 2 ⇒ range = 8
    ideal_range = 10 * (1 - 0.0) - 10 * (1 - 0.8)  # = 8
    res = {}
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        mean_0 = sub[sub["level"] == 0.0]["score"].mean()
        mean_08 = sub[sub["level"] == 0.8]["score"].mean()
        observed_range = mean_0 - mean_08
        compression = observed_range / ideal_range
        ceiling_ratio = mean_0 / 10.0
        floor_ratio = mean_08 / 0.0 if False else mean_08  # absolute floor
        info = {
            "mean_at_0.0": float(mean_0), "mean_at_0.8": float(mean_08),
            "observed_range": float(observed_range),
            "ideal_range": float(ideal_range),
            "compression_ratio": float(compression),
            "ceiling_ratio": float(ceiling_ratio),
            "floor_absolute": float(mean_08),
        }
        res[model] = info
        print(f"\n  {model}:")
        print(f"    Mean at level 0.0: {mean_0:.2f}  (ceiling ratio: {ceiling_ratio:.3f})")
        print(f"    Mean at level 0.8: {mean_08:.2f}")
        print(f"    Observed range: {observed_range:.2f}  Ideal range: {ideal_range:.1f}")
        print(f"    Compression ratio: {compression:.3f}  "
              f"(1.0 = perfect calibration, <1.0 = compressed)")
    RESULTS["T3"] = res


# ═══════════════════════════════════════════════════════════════
# T4: BOOTSTRAP CI FOR COMPRESSION RATIO
# ═══════════════════════════════════════════════════════════════

def test_t4(df):
    header("T4: Bootstrap CI for Compression Ratio (10K iterations)")
    ideal_range = 8.0
    n_boot = 10000
    rng = np.random.RandomState(42)
    res = {}
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        # Article-level means at level 0 and 0.8
        art_0 = sub[sub["level"] == 0.0].groupby("article")["score"].mean()
        art_08 = sub[sub["level"] == 0.8].groupby("article")["score"].mean()
        articles = art_0.index.values
        vals_0 = art_0.values
        vals_08 = art_08.reindex(articles).values

        boot_ratios = np.empty(n_boot)
        n = len(articles)
        for i in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot_ratios[i] = (vals_0[idx].mean() - vals_08[idx].mean()) / ideal_range

        ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])
        point = (vals_0.mean() - vals_08.mean()) / ideal_range
        info = {"point_estimate": float(point),
                "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi),
                "excludes_1": bool(ci_hi < 1.0)}
        res[model] = info
        print(f"\n  {model}:")
        print(f"    Compression ratio: {point:.3f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"    CI excludes 1.0 (perfect calibration): {info['excludes_1']}")
        if info["excludes_1"]:
            print(f"    ⇒ Compression is statistically confirmed")
    RESULTS["T4"] = res


# ═══════════════════════════════════════════════════════════════
# T5: LINEAR REGRESSION PER AXIS PER MODEL
# ═══════════════════════════════════════════════════════════════

def test_t5(df):
    header("T5: Linear Regression (score ~ level) per axis per model")
    res = {}
    for model in MODEL_NAMES:
        res[model] = {}
        print(f"\n  {model}:")
        for axis in AXES_ORDER:
            sub = df[(df["model"] == model) & (df["axis"] == axis)]
            r = sp_stats.linregress(sub["level"], sub["score"])
            info = {"slope": float(r.slope), "intercept": float(r.intercept),
                    "r_squared": float(r.rvalue ** 2), "p_value": float(r.pvalue),
                    "stderr": float(r.stderr)}
            res[model][axis] = info
            print(f"    {axis:12s}  β₁={r.slope:+.3f}  β₀={r.intercept:.2f}  "
                  f"R²={r.rvalue**2:.3f}  p={r.pvalue:.2e}  SE={r.stderr:.3f}")
    RESULTS["T5"] = res
    return res


# ═══════════════════════════════════════════════════════════════
# T6: NON-LINEARITY TEST (linear vs quadratic)
# ═══════════════════════════════════════════════════════════════

def test_t6(df):
    header("T6: Non-linearity Test (F-test: linear vs quadratic)")
    import statsmodels.api as sm
    res = {}
    for model in MODEL_NAMES:
        res[model] = {}
        print(f"\n  {model}:")
        for axis in AXES_ORDER:
            sub = df[(df["model"] == model) & (df["axis"] == axis)]
            y = sub["score"].values
            x = sub["level"].values

            # Linear model
            X_lin = sm.add_constant(x)
            fit_lin = sm.OLS(y, X_lin).fit()

            # Quadratic model
            X_quad = sm.add_constant(np.column_stack([x, x ** 2]))
            fit_quad = sm.OLS(y, X_quad).fit()

            # F-test for the quadratic term
            ss_lin = fit_lin.ssr
            ss_quad = fit_quad.ssr
            df_diff = fit_quad.df_model - fit_lin.df_model
            df_res = fit_quad.df_resid
            f_stat = ((ss_lin - ss_quad) / df_diff) / (ss_quad / df_res)
            p_val = 1 - sp_stats.f.cdf(f_stat, df_diff, df_res)
            quad_coef = fit_quad.params[2]

            info = {"f_stat": float(f_stat), "p_value": float(p_val),
                    "quadratic_coef": float(quad_coef),
                    "significant": bool(p_val < 0.05)}
            res[model][axis] = info
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
                  "*" if p_val < 0.05 else "ns"
            print(f"    {axis:12s}  F={f_stat:.2f}  p={p_val:.4f} {sig}  "
                  f"quad_coef={quad_coef:+.2f}")
    RESULTS["T6"] = res


# ═══════════════════════════════════════════════════════════════
# T7: KENDALL'S TAU (monotonicity)
# ═══════════════════════════════════════════════════════════════

def test_t7(df):
    header("T7: Kendall's τ (monotonicity)")
    res = {}
    for model in MODEL_NAMES:
        res[model] = {}
        print(f"\n  {model}:")
        for axis in AXES_ORDER:
            sub = df[(df["model"] == model) & (df["axis"] == axis)]
            tau, p = sp_stats.kendalltau(sub["level"], sub["score"])
            info = {"tau": float(tau), "p_value": float(p)}
            res[model][axis] = info
            print(f"    {axis:12s}  τ={tau:+.3f}  p={p:.2e}")
    RESULTS["T7"] = res


# ═══════════════════════════════════════════════════════════════
# T8: MIXED-EFFECTS MODEL
# ═══════════════════════════════════════════════════════════════

def test_t8(df):
    header("T8: Linear Mixed-Effects Model")
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  [SKIP] statsmodels not installed")
        RESULTS["T8"] = {"error": "statsmodels not installed"}
        return

    print("  Fitting: score ~ level * C(axis) * C(model) + (1 + level | article)")
    print("  This may take a minute...")

    # Try random slopes first, fall back to random intercept
    try:
        md = smf.mixedlm("score ~ level * C(axis) * C(model)",
                          df, groups=df["article"], re_formula="~level")
        fit = md.fit(reml=True, maxiter=200)
        re_type = "random intercept + slope"
    except Exception as e:
        print(f"  [WARN] Random slopes failed ({e}), falling back to random intercept only")
        md = smf.mixedlm("score ~ level * C(axis) * C(model)",
                          df, groups=df["article"])
        fit = md.fit(reml=True, maxiter=200)
        re_type = "random intercept only"

    print(f"\n  Random effects: {re_type}")
    print(f"  Converged: {fit.converged}")
    print(f"  Log-likelihood: {fit.llf:.1f}")
    print(f"  AIC: {fit.aic:.1f}  BIC: {fit.bic:.1f}")
    print(f"\n  Fixed effects:")

    fe = {}
    for name, coef in fit.fe_params.items():
        se = fit.bse_fe[name]
        z = fit.tvalues[name]
        p = fit.pvalues[name]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        short_name = name.replace("C(axis)[T.", "").replace("C(model)[T.", "") \
                         .replace("]", "")
        fe[name] = {"coef": float(coef), "se": float(se),
                    "z": float(z), "p": float(p)}
        print(f"    {short_name:45s}  β={coef:+.3f}  SE={se:.3f}  "
              f"z={z:+.2f}  p={p:.4f} {sig}")

    # Random effects variance
    re_var = {"group_var": float(fit.cov_re.iloc[0, 0])}
    if fit.cov_re.shape[0] > 1:
        re_var["slope_var"] = float(fit.cov_re.iloc[1, 1])
    print(f"\n  Random effects variance: {re_var}")

    RESULTS["T8"] = {"fixed_effects": fe, "random_effects": re_var,
                     "re_type": re_type, "converged": bool(fit.converged),
                     "aic": float(fit.aic), "bic": float(fit.bic)}
    return fit


# ═══════════════════════════════════════════════════════════════
# T9: POST-HOC PAIRWISE AXIS COMPARISONS
# ═══════════════════════════════════════════════════════════════

def test_t9(df):
    header("T9: Post-hoc Axis Pairwise Comparisons (Wilcoxon)")
    res = {}
    for model in MODEL_NAMES:
        res[model] = {}
        print(f"\n  {model}:")
        sub = df[df["model"] == model]
        for i, ax1 in enumerate(AXES_ORDER):
            for ax2 in AXES_ORDER[i + 1:]:
                s1 = sub[sub["axis"] == ax1]["score"].values
                s2 = sub[sub["axis"] == ax2]["score"].values
                stat, p = sp_stats.mannwhitneyu(s1, s2, alternative="two-sided")
                d = cliffs_delta(s1, s2)
                key = f"{ax1}_vs_{ax2}"
                res[model][key] = {"U": float(stat), "p": float(p),
                                   "cliffs_delta": float(d)}
                sig = "***" if p < 0.001 else "**" if p < 0.01 else \
                      "*" if p < 0.05 else "ns"
                print(f"    {ax1:11s} vs {ax2:11s}  U={stat:.0f}  p={p:.4f} {sig}  "
                      f"Cliff's δ={d:+.3f}")
    RESULTS["T9"] = res


# ═══════════════════════════════════════════════════════════════
# T10: LEVENE'S TEST (variance homogeneity across levels)
# ═══════════════════════════════════════════════════════════════

def test_t10(df):
    header("T10: Levene's Test (variance homogeneity across levels)")
    res = {}
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        groups = [sub[sub["level"] == lv]["score"].values for lv in LEVELS]
        stat, p = sp_stats.levene(*groups)
        variances = {f"level_{lv}": float(np.var(g, ddof=1)) for lv, g in zip(LEVELS, groups)}
        info = {"statistic": float(stat), "p_value": float(p),
                "variances_by_level": variances}
        res[model] = info
        print(f"\n  {model}:")
        print(f"    Levene's W={stat:.2f}  p={p:.4f}")
        for lv, v in variances.items():
            print(f"      {lv}: var={v:.3f}")
    RESULTS["T10"] = res


# ═══════════════════════════════════════════════════════════════
# T11: PEARSON + SPEARMAN (inter-model correlation)
# ═══════════════════════════════════════════════════════════════

def test_t11(paired):
    header("T11: Inter-model Correlation (Pearson + Spearman)")
    g, e = paired["gpt_score"].values, paired["gemini_score"].values
    pearson_r, pearson_p = sp_stats.pearsonr(g, e)
    spearman_r, spearman_p = sp_stats.spearmanr(g, e)
    info = {"pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p)}
    RESULTS["T11"] = info
    print(f"\n  Pearson  r = {pearson_r:.4f}  p = {pearson_p:.2e}")
    print(f"  Spearman ρ = {spearman_r:.4f}  p = {spearman_p:.2e}")


# ═══════════════════════════════════════════════════════════════
# T12: WILCOXON SIGNED-RANK (inter-model systematic bias)
# ═══════════════════════════════════════════════════════════════

def test_t12(paired):
    header("T12: Wilcoxon Signed-Rank (Gemini − GPT systematic bias)")
    d = paired["diff"].values
    stat, p, r = rank_biserial(paired["gemini_score"].values,
                               paired["gpt_score"].values)
    median_diff = float(np.median(d))
    mean_diff = float(np.mean(d))
    info = {"wilcoxon_stat": float(stat), "p_value": float(p),
            "rank_biserial_r": float(r),
            "median_diff": median_diff, "mean_diff": mean_diff}
    RESULTS["T12"] = info
    print(f"\n  Median difference (Gemini − GPT): {median_diff:+.2f}")
    print(f"  Mean difference:                  {mean_diff:+.2f}")
    print(f"  Wilcoxon W = {stat:.0f}  p = {p:.2e}")
    print(f"  Rank-biserial r = {r:+.3f}")
    direction = "Gemini scores higher" if mean_diff > 0 else "GPT scores higher"
    print(f"  ⇒ {direction}")


# ═══════════════════════════════════════════════════════════════
# T13: PER-AXIS MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════

def test_t13(paired):
    header("T13: Per-axis Model Comparison (Wilcoxon signed-rank)")
    res = {}
    for axis in AXES_ORDER:
        sub = paired[paired["axis"] == axis]
        stat, p, r = rank_biserial(sub["gemini_score"].values,
                                   sub["gpt_score"].values)
        median_d = float(np.median(sub["diff"]))
        res[axis] = {"wilcoxon_stat": float(stat), "p_value": float(p),
                     "rank_biserial_r": float(r), "median_diff": median_d}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else \
              "*" if p < 0.05 else "ns"
        print(f"  {axis:12s}  median_diff={median_d:+.1f}  W={stat:.0f}  "
              f"p={p:.4f} {sig}  r={r:+.3f}")
    RESULTS["T13"] = res


# ═══════════════════════════════════════════════════════════════
# T14: INTRA-CONDITION RELIABILITY (ICC)
# ═══════════════════════════════════════════════════════════════

def test_t14(df):
    header("T14: Intra-condition Reliability (ICC + CV across 3 reps)")
    res = {}
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        print(f"\n  {model}:")

        # Overall ICC
        groups = []
        cvs = []
        for (art, axis, lv), g in sub.groupby(["article", "axis", "level"]):
            scores = sorted(g["score"].values)
            if len(scores) == 3:
                groups.append(scores)
                m = np.mean(scores)
                if m > 0:
                    cvs.append(np.std(scores, ddof=1) / m)

        icc = compute_icc_oneway(groups)
        mean_cv = float(np.mean(cvs)) if cvs else 0
        print(f"    Overall ICC(1,1) = {icc:.3f}   Mean CV = {mean_cv:.3f}")

        # By axis
        icc_by_axis = {}
        cv_by_axis = {}
        for axis in AXES_ORDER:
            ax_groups, ax_cvs = [], []
            for (art, lv), g in sub[sub["axis"] == axis].groupby(["article", "level"]):
                scores = sorted(g["score"].values)
                if len(scores) == 3:
                    ax_groups.append(scores)
                    m = np.mean(scores)
                    if m > 0:
                        ax_cvs.append(np.std(scores, ddof=1) / m)
            ax_icc = compute_icc_oneway(ax_groups) if ax_groups else 0
            ax_cv = float(np.mean(ax_cvs)) if ax_cvs else 0
            icc_by_axis[axis] = ax_icc
            cv_by_axis[axis] = ax_cv
            print(f"      {axis:12s}  ICC={ax_icc:.3f}  CV={ax_cv:.3f}")

        # By level
        icc_by_level = {}
        cv_by_level = {}
        for lv in LEVELS:
            lv_groups, lv_cvs = [], []
            for (art, axis), g in sub[sub["level"] == lv].groupby(["article", "axis"]):
                scores = sorted(g["score"].values)
                if len(scores) == 3:
                    lv_groups.append(scores)
                    m = np.mean(scores)
                    if m > 0:
                        lv_cvs.append(np.std(scores, ddof=1) / m)
            lv_icc = compute_icc_oneway(lv_groups) if lv_groups else 0
            lv_cv = float(np.mean(lv_cvs)) if lv_cvs else 0
            icc_by_level[str(lv)] = lv_icc
            cv_by_level[str(lv)] = lv_cv
            print(f"      level {lv:.1f}      ICC={lv_icc:.3f}  CV={lv_cv:.3f}")

        res[model] = {"overall_icc": float(icc), "mean_cv": mean_cv,
                      "icc_by_axis": {k: float(v) for k, v in icc_by_axis.items()},
                      "cv_by_axis": {k: float(v) for k, v in cv_by_axis.items()},
                      "icc_by_level": {k: float(v) for k, v in icc_by_level.items()},
                      "cv_by_level": {k: float(v) for k, v in cv_by_level.items()}}
    RESULTS["T14"] = res


# ═══════════════════════════════════════════════════════════════
# T15: ARTICLE LENGTH AS COVARIATE
# ═══════════════════════════════════════════════════════════════

def test_t15(df):
    header("T15: Article Length as Covariate")
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  [SKIP] statsmodels not installed")
        return

    res = {}
    for model in MODEL_NAMES:
        sub = df[df["model"] == model].copy()
        sub["length_z"] = (sub["article_length"] - sub["article_length"].mean()) / \
                          sub["article_length"].std()
        md = smf.mixedlm("score ~ level * C(axis) + length_z",
                          sub, groups=sub["article"])
        fit = md.fit(reml=True)
        coef = fit.fe_params["length_z"]
        p = fit.pvalues["length_z"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        info = {"length_coef": float(coef), "p_value": float(p),
                "significant": bool(p < 0.05)}
        res[model] = info
        print(f"\n  {model}:")
        print(f"    Length (z-scored) coefficient: {coef:+.4f}  p={p:.4f} {sig}")
        if not info["significant"]:
            print(f"    ⇒ Article length is NOT a significant confound")
    RESULTS["T15"] = res


# ═══════════════════════════════════════════════════════════════
# T16: CATEGORY EFFECTS
# ═══════════════════════════════════════════════════════════════

def test_t16(df):
    header("T16: Category Effects (Kruskal-Wallis on undegraded scores)")
    res = {}
    for model in MODEL_NAMES:
        sub = df[(df["model"] == model) & (df["level"] == 0.0)]
        cats = sub["category"].unique()
        groups = [sub[sub["category"] == c]["score"].values for c in cats]
        stat, p = sp_stats.kruskal(*groups)
        cat_means = {c: float(sub[sub["category"] == c]["score"].mean()) for c in sorted(cats)}
        info = {"H_statistic": float(stat), "p_value": float(p),
                "n_categories": int(len(cats)), "category_means": cat_means}
        res[model] = info
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n  {model}:")
        print(f"    H={stat:.2f}  p={p:.4f} {sig}  ({len(cats)} categories)")
        if p < 0.05:
            best = max(cat_means, key=cat_means.get)
            worst = min(cat_means, key=cat_means.get)
            print(f"    ⇒ Significant category effect!")
            print(f"      Highest: {best} ({cat_means[best]:.2f})")
            print(f"      Lowest:  {worst} ({cat_means[worst]:.2f})")
        else:
            print(f"    ⇒ No significant category bias")
    RESULTS["T16"] = res


# ═══════════════════════════════════════════════════════════════
# T17: EFFECT SIZES (Cliff's delta + Cohen's d)
# ═══════════════════════════════════════════════════════════════

def test_t17(df, paired):
    header("T17: Effect Sizes (Cliff's δ primary, Cohen's d secondary)")
    res = {}

    # Within-model: undegraded vs max-degraded (independent)
    print("\n  Within-model (undegraded vs max-degraded):")
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        und = sub[sub["level"] == 0.0]["score"].values
        deg = sub[sub["level"] == 0.8]["score"].values
        cd = cliffs_delta(und, deg)
        d = cohens_d(und, deg)
        key = f"{model}_0v08"
        res[key] = {"cliffs_delta": float(cd), "cohens_d": float(d)}
        print(f"    {model}:  Cliff's δ = {cd:+.3f}  Cohen's d = {d:+.3f}")

    # Between-model: at each level (paired)
    print("\n  Between-model (Gemini − GPT) at each level:")
    for lv in LEVELS:
        sub = paired[paired["level"] == lv]
        stat, p, r = rank_biserial(sub["gemini_score"].values,
                                   sub["gpt_score"].values)
        d = cohens_d(sub["gemini_score"].values, sub["gpt_score"].values)
        key = f"inter_model_level_{lv}"
        res[key] = {"rank_biserial_r": float(r), "cohens_d": float(d), "p": float(p)}
        print(f"    Level {lv:.1f}:  rank-biserial r = {r:+.3f}  Cohen's d = {d:+.3f}  p={p:.4f}")

    RESULTS["T17"] = res


# ═══════════════════════════════════════════════════════════════
# T18: BENJAMINI-HOCHBERG FDR CORRECTION
# ═══════════════════════════════════════════════════════════════

def test_t18():
    header("T18: Benjamini-Hochberg FDR Correction")

    # Collect all p-values from previous tests
    pvals = []

    def add_p(test, label, p):
        pvals.append({"test": test, "label": label, "raw_p": float(p)})

    # T5: regression p-values
    if "T5" in RESULTS:
        for model in MODEL_NAMES:
            for axis in AXES_ORDER:
                add_p("T5", f"{model}_{axis}", RESULTS["T5"][model][axis]["p_value"])

    # T6: nonlinearity
    if "T6" in RESULTS:
        for model in MODEL_NAMES:
            for axis in AXES_ORDER:
                add_p("T6", f"{model}_{axis}", RESULTS["T6"][model][axis]["p_value"])

    # T7: Kendall
    if "T7" in RESULTS:
        for model in MODEL_NAMES:
            for axis in AXES_ORDER:
                add_p("T7", f"{model}_{axis}", RESULTS["T7"][model][axis]["p_value"])

    # T9: post-hoc
    if "T9" in RESULTS:
        for model in MODEL_NAMES:
            for key, v in RESULTS["T9"][model].items():
                add_p("T9", f"{model}_{key}", v["p"])

    # T10: Levene
    if "T10" in RESULTS:
        for model in MODEL_NAMES:
            add_p("T10", f"{model}", RESULTS["T10"][model]["p_value"])

    # T11: correlation
    if "T11" in RESULTS:
        add_p("T11", "pearson", RESULTS["T11"]["pearson_p"])
        add_p("T11", "spearman", RESULTS["T11"]["spearman_p"])

    # T12: Wilcoxon
    if "T12" in RESULTS:
        add_p("T12", "overall", RESULTS["T12"]["p_value"])

    # T13: per-axis
    if "T13" in RESULTS:
        for axis in AXES_ORDER:
            add_p("T13", axis, RESULTS["T13"][axis]["p_value"])

    # T15: length
    if "T15" in RESULTS:
        for model in MODEL_NAMES:
            if model in RESULTS["T15"]:
                add_p("T15", model, RESULTS["T15"][model]["p_value"])

    # T16: category
    if "T16" in RESULTS:
        for model in MODEL_NAMES:
            add_p("T16", model, RESULTS["T16"][model]["p_value"])

    if not pvals:
        print("  No p-values to correct")
        return

    # Sort by p-value
    pvals.sort(key=lambda x: x["raw_p"])
    m = len(pvals)
    for i, pv in enumerate(pvals):
        pv["bh_adjusted_p"] = min(1.0, pv["raw_p"] * m / (i + 1))

    # Enforce monotonicity (from bottom up)
    for i in range(m - 2, -1, -1):
        pvals[i]["bh_adjusted_p"] = min(pvals[i]["bh_adjusted_p"],
                                        pvals[i + 1]["bh_adjusted_p"])

    print(f"\n  Total p-values corrected: {m}")
    print(f"\n  {'Test':6s} {'Label':35s} {'Raw p':>12s} {'BH adj p':>12s} {'Sig':>5s}")
    print(f"  {'-'*75}")
    for pv in pvals:
        sig = "***" if pv["bh_adjusted_p"] < 0.001 else \
              "**" if pv["bh_adjusted_p"] < 0.01 else \
              "*" if pv["bh_adjusted_p"] < 0.05 else "ns"
        print(f"  {pv['test']:6s} {pv['label']:35s} {pv['raw_p']:12.2e} "
              f"{pv['bh_adjusted_p']:12.2e} {sig:>5s}")

    n_sig = sum(1 for p in pvals if p["bh_adjusted_p"] < 0.05)
    print(f"\n  Significant after BH correction: {n_sig}/{m}")

    RESULTS["T18"] = pvals


# ═══════════════════════════════════════════════════════════════
# T19: COHEN'S WEIGHTED KAPPA (supplementary)
# ═══════════════════════════════════════════════════════════════

def test_t19(paired):
    header("T19: Cohen's Weighted κ (supplementary)")
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("  [SKIP] sklearn not installed")
        return

    g = paired["gpt_score"].values
    e = paired["gemini_score"].values
    kappa_linear = cohen_kappa_score(g, e, weights="linear")
    kappa_quad = cohen_kappa_score(g, e, weights="quadratic")
    info = {"kappa_linear": float(kappa_linear),
            "kappa_quadratic": float(kappa_quad)}
    RESULTS["T19"] = info
    print(f"\n  Weighted κ (linear weights):    {kappa_linear:.4f}")
    print(f"  Weighted κ (quadratic weights): {kappa_quad:.4f}")


# ═══════════════════════════════════════════════════════════════
# GRAPHS
# ═══════════════════════════════════════════════════════════════

def graph_g1(df):
    """G1: Dose-response curves — one line per model per axis."""
    header("G1: Dose-response curves")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Distinct color + marker combos so 11 lines stay separable
    _markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
    _cmap = plt.cm.tab20
    _nmod = len(MODEL_NAMES)
    _line_colors = [_cmap(i / _nmod) for i in range(_nmod)]

    for i, axis in enumerate(AXES_ORDER):
        ax = axes[i]
        for mi, model in enumerate(MODEL_NAMES):
            sub = df[(df["model"] == model) & (df["axis"] == axis)]
            means, ci_lo, ci_hi = [], [], []
            for lv in LEVELS:
                lv_sub = sub[sub["level"] == lv]
                m = lv_sub["score"].mean()
                lo, hi = block_bootstrap_means(lv_sub, n_boot=5000)
                means.append(m)
                ci_lo.append(lo)
                ci_hi.append(hi)
            col = _line_colors[mi]
            mk = _markers[mi % len(_markers)]
            ax.plot(LEVELS, means, marker=mk, linestyle="-", color=col,
                    label=model, lw=1.8, ms=6, markeredgewidth=0.8,
                    markeredgecolor="white")
            ax.fill_between(LEVELS, ci_lo, ci_hi, alpha=0.10, color=col)

        perfect = [10 * (1 - lv) for lv in LEVELS]
        ax.plot(LEVELS, perfect, "--", color="black", alpha=0.35, lw=1.2,
                label="Perfect calibration")

        ax.set_title(AXIS_LABELS[axis], fontweight="bold")
        ax.set_ylim(-0.5, 10.5)
        ax.set_ylabel("LLM Score" if i % 2 == 0 else "")
        ax.set_xlabel("Degradation Level" if i >= 2 else "")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", fontsize=8, ncol=2,
                      framealpha=0.85, borderpad=0.5)

    fig.suptitle("Dose-Response: LLM Score vs Degradation Level (per model)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G1_dose_response.png")


def graph_g1b(df):
    """G1b: Grand-average dose-response — one line per axis, averaged across all models."""
    header("G1b: Grand-average dose-response (all models combined)")
    _axis_colors = {"grammar": "#1f77b4", "coherence": "#ff7f0e",
                    "information": "#2ca02c", "lexical": "#d62728"}
    _axis_markers = {"grammar": "o", "coherence": "s",
                     "information": "^", "lexical": "D"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for axis in AXES_ORDER:
        sub = df[df["axis"] == axis]
        means, ci_lo, ci_hi = [], [], []
        for lv in LEVELS:
            lv_sub = sub[sub["level"] == lv]["score"]
            m = float(lv_sub.mean())
            # 95% CI via SEM across the pooled distribution
            se = float(lv_sub.sem())
            means.append(m)
            ci_lo.append(m - 1.96 * se)
            ci_hi.append(m + 1.96 * se)
        col = _axis_colors[axis]
        mk = _axis_markers[axis]
        ax.plot(LEVELS, means, marker=mk, linestyle="-", color=col,
                label=AXIS_LABELS[axis], lw=2.2, ms=8,
                markeredgewidth=0.8, markeredgecolor="white")
        ax.fill_between(LEVELS, ci_lo, ci_hi, alpha=0.15, color=col)
        # Annotate the mean value at each level
        for lv, m in zip(LEVELS, means):
            ax.annotate(f"{m:.2f}", xy=(lv, m),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5, color=col)

    perfect = [10 * (1 - lv) for lv in LEVELS]
    ax.plot(LEVELS, perfect, "--", color="black", alpha=0.4, lw=1.5,
            label="Perfect calibration")

    n_models = len(MODEL_NAMES)
    ax.set_xlabel("Degradation Level", fontsize=12)
    ax.set_ylabel("Mean LLM Score (avg across all models)", fontsize=12)
    ax.set_title(
        f"Grand-Average Dose-Response by Axis\n"
        f"(averaged across {n_models} models, ±95% CI)",
        fontweight="bold", fontsize=13)
    ax.set_ylim(-0.5, 10.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "G1b_grand_avg_dose_response.png")


def graph_g2(df):
    """G2: Cross-axis comparison per model."""
    header("G2: Cross-axis comparison")
    _n = len(MODEL_NAMES)
    _ncols = min(_n, 3)
    _nrows = math.ceil(_n / _ncols)
    fig, _axes_arr = plt.subplots(_nrows, _ncols, figsize=(7 * _ncols, 5 * _nrows), sharey=True, squeeze=False)
    _axes_flat = [_axes_arr[r][c] for r in range(_nrows) for c in range(_ncols)]
    for j, model in enumerate(MODEL_NAMES):
        ax = _axes_flat[j]
        for axis in AXES_ORDER:
            sub = df[(df["model"] == model) & (df["axis"] == axis)]
            means = [sub[sub["level"] == lv]["score"].mean() for lv in LEVELS]
            ax.plot(LEVELS, means, "o-", label=AXIS_LABELS[axis], lw=2, ms=5)
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Mean LLM Score" if j % _ncols == 0 else "")
        ax.set_ylim(-0.5, 10.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    for j in range(_n, _nrows * _ncols):
        _axes_flat[j].set_visible(False)
    fig.suptitle("Cross-Axis Sensitivity Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G2_cross_axis.png")


def graph_g3(df):
    """G3: Overall score histograms."""
    header("G3: Overall score histograms")
    _n = len(MODEL_NAMES)
    _ncols = min(_n, 3)
    _nrows = math.ceil(_n / _ncols)
    fig, _axes_arr = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 5 * _nrows), sharey=True, squeeze=False)
    _axes_flat = [_axes_arr[r][c] for r in range(_nrows) for c in range(_ncols)]
    for j, model in enumerate(MODEL_NAMES):
        ax = _axes_flat[j]
        s = df[df["model"] == model]["score"]
        counts = s.value_counts().reindex(range(11), fill_value=0)
        bars = ax.bar(range(11), counts, color=COLORS[model], edgecolor="white", alpha=0.85)
        # Annotate dead zones
        for val in [0, 10]:
            if counts[val] == 0:
                ax.annotate(f"0", xy=(val, 0), xytext=(val, counts.max() * 0.1),
                            ha="center", fontsize=9, color="red",
                            arrowprops=dict(arrowstyle="->", color="red"))
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if j % _ncols == 0 else "")
        ax.set_xticks(range(11))
    for j in range(_n, _nrows * _ncols):
        _axes_flat[j].set_visible(False)
    fig.suptitle("Score Distribution (n=9,000 per model)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G3_score_histograms.png")


def graph_g4(df):
    """G4: Undegraded (level=0.0) distribution."""
    header("G4: Undegraded distribution")
    _n = len(MODEL_NAMES)
    width = 0.8 / _n
    fig, ax = plt.subplots(figsize=(max(8, _n * 2), 5))
    und = df[df["level"] == 0.0]
    for j, model in enumerate(MODEL_NAMES):
        s = und[und["model"] == model]["score"]
        counts = s.value_counts().reindex(range(11), fill_value=0)
        offset = -0.4 + j * width + width / 2
        ax.bar(np.arange(11) + offset, counts, width, label=model,
               color=COLORS[model], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution for Pristine Texts (level = 0.0, n=1,800 per model)",
                 fontweight="bold")
    ax.set_xticks(range(11))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "G4_undegraded_dist.png")


def graph_g5(df):
    """G5: Per-level distributions (faceted density)."""
    header("G5: Per-level distributions")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for i, lv in enumerate(LEVELS):
        ax = axes[i]
        _n5 = len(MODEL_NAMES)
        _bw = 0.8 / _n5
        for jj, model in enumerate(MODEL_NAMES):
            s = df[(df["model"] == model) & (df["level"] == lv)]["score"]
            counts = s.value_counts().reindex(range(11), fill_value=0) / len(s)
            _off = -0.4 + jj * _bw + _bw / 2
            ax.bar(np.arange(11) + _off, counts, _bw, color=COLORS[model], alpha=0.8, label=model)
        ax.set_title(f"Level {lv:.1f}", fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_xticks(range(0, 11, 2))
        if i == 0:
            ax.set_ylabel("Proportion")
            ax.legend(fontsize=8)
    fig.suptitle("Score Distribution by Degradation Level", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G5_per_level_dist.png")


def graph_g6(df):
    """G6: Boxplots axis × level grid."""
    header("G6: Boxplots (axis × level)")
    _n = len(MODEL_NAMES)
    fig, axes = plt.subplots(_n, 4, figsize=(16, 4 * _n), sharey=True, squeeze=False)
    for row, model in enumerate(MODEL_NAMES):
        for col, axis in enumerate(AXES_ORDER):
            ax = axes[row, col]
            sub = df[(df["model"] == model) & (df["axis"] == axis)]
            data = [sub[sub["level"] == lv]["score"].values for lv in LEVELS]
            bp = ax.boxplot(data, tick_labels=[f"{lv:.1f}" for lv in LEVELS],
                            patch_artist=True, medianprops=dict(color="black"))
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS[model])
                patch.set_alpha(0.6)
            if row == 0:
                ax.set_title(AXIS_LABELS[axis], fontweight="bold")
            if col == 0:
                ax.set_ylabel(model, fontsize=11, fontweight="bold")
            if row == len(MODEL_NAMES) - 1:
                ax.set_xlabel("Level")
            ax.set_ylim(-0.5, 10.5)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Score Distributions by Axis × Level", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G6_boxplots_grid.png")


def graph_g7(df):
    """G7: Violin plots per level."""
    header("G7: Violin plots per level")
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = []
    violins_data = []
    labels = []
    _n = len(MODEL_NAMES)
    _spacing = _n + 1
    for i, lv in enumerate(LEVELS):
        for j, model in enumerate(MODEL_NAMES):
            pos = i * _spacing + j
            s = df[(df["model"] == model) & (df["level"] == lv)]["score"].values
            positions.append(pos)
            violins_data.append(s)
            labels.append(f"{lv:.1f}\n{model.split()[0]}")

    parts = ax.violinplot(violins_data, positions=positions, showmeans=True,
                          showmedians=True, widths=0.8)
    for i, pc in enumerate(parts["bodies"]):
        model = MODEL_NAMES[i % _n]
        pc.set_facecolor(COLORS[model])
        pc.set_alpha(0.6)

    ax.set_xticks([i * _spacing + (_n - 1) / 2 for i in range(5)])
    ax.set_xticklabels([f"{lv:.1f}" for lv in LEVELS])
    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("Score")
    ax.set_title("Score Distributions (Violin) by Level and Model", fontweight="bold")
    ax.set_ylim(-0.5, 10.5)
    ax.grid(axis="y", alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLORS[m], alpha=0.6, label=m) for m in MODEL_NAMES])
    fig.tight_layout()
    savefig(fig, "G7_violins.png")


def graph_g8(df):
    """G8: Forest plot of regression slopes with CIs."""
    header("G8: Forest plot (sensitivity slopes)")
    if "T5" not in RESULTS:
        print("  [SKIP] T5 not run")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = []
    y_labels = []
    pos = 0
    for axis in AXES_ORDER:
        for model in MODEL_NAMES:
            r = RESULTS["T5"][model][axis]
            slope = r["slope"]
            se = r["stderr"]
            ci_lo = slope - 1.96 * se
            ci_hi = slope + 1.96 * se
            ax.errorbar(slope, pos, xerr=[[slope - ci_lo], [ci_hi - slope]],
                        fmt="o", color=COLORS[model], capsize=4, ms=8, lw=2)
            y_pos.append(pos)
            y_labels.append(f"{AXIS_LABELS[axis]} / {model}")
            pos += 1
        pos += 0.5  # gap between axes

    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Regression Slope (β₁)")
    ax.set_title("Sensitivity: Regression Slopes ± 95% CI", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, "G8_forest_plot.png")


def graph_g9(df):
    """G9: Calibration plot."""
    header("G9: Calibration plot")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration line
    perfect = [10 * (1 - lv) for lv in LEVELS]
    ax.plot(LEVELS, perfect, "k--", lw=2, label="Perfect calibration", alpha=0.5)

    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        means = [sub[sub["level"] == lv]["score"].mean() for lv in LEVELS]
        ax.plot(LEVELS, means, "o-", color=COLORS[model], label=model, lw=2.5, ms=8)
        # Shade compression gap
        ax.fill_between(LEVELS, means, perfect, alpha=0.08, color=COLORS[model])

    ax.set_xlabel("Degradation Level", fontsize=12)
    ax.set_ylabel("Mean LLM Score", fontsize=12)
    ax.set_title("Calibration: Observed vs Perfect Scoring", fontweight="bold", fontsize=13)
    ax.set_ylim(-0.5, 10.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate("Compression\ngap", xy=(0.4, 7), fontsize=10, color="gray",
                ha="center", fontstyle="italic")
    fig.tight_layout()
    savefig(fig, "G9_calibration.png")


def graph_g10(paired):
    """G10: Inter-model scatter."""
    header("G10: Inter-model scatter")
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(5, 5)
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # Jitter
    rng = np.random.RandomState(42)
    jitter = 0.15
    x = paired["gpt_score"].values + rng.uniform(-jitter, jitter, len(paired))
    y = paired["gemini_score"].values + rng.uniform(-jitter, jitter, len(paired))

    # Color by level
    level_colors = {0.0: "#2ca02c", 0.2: "#1f77b4", 0.4: "#ff7f0e",
                    0.6: "#d62728", 0.8: "#9467bd"}
    for lv in LEVELS:
        mask = paired["level"] == lv
        ax_main.scatter(x[mask], y[mask], s=3, alpha=0.3, color=level_colors[lv],
                        label=f"Level {lv:.1f}", rasterized=True)

    ax_main.plot([-0.5, 10.5], [-0.5, 10.5], "k--", alpha=0.4, lw=1)
    ax_main.set_xlabel("GPT-5 mini Score", fontsize=12)
    ax_main.set_ylabel("Gemini 3 Flash Score", fontsize=12)
    ax_main.set_xlim(-0.5, 10.5)
    ax_main.set_ylim(-0.5, 10.5)
    ax_main.legend(loc="upper left", fontsize=8, markerscale=3)
    ax_main.grid(True, alpha=0.2)

    # Marginals
    ax_top.hist(paired["gpt_score"], bins=np.arange(-0.5, 11.5, 1),
                color=COLORS.get("gpt-5-mini", "#2171b5"), alpha=0.7, edgecolor="white")
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_right.hist(paired["gemini_score"], bins=np.arange(-0.5, 11.5, 1),
                  color=COLORS.get("gemini-3-flash", "#e6550d"), alpha=0.7, edgecolor="white",
                  orientation="horizontal")
    ax_right.set_xlabel("Count")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle("Inter-Model Agreement (n=9,000 paired samples)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G10_inter_model_scatter.png")


def graph_g11(paired):
    """G11: Paired difference histogram."""
    header("G11: Paired difference histogram")
    fig, ax = plt.subplots(figsize=(8, 5))
    d = paired["diff"]
    counts = d.value_counts().reindex(range(-9, 10), fill_value=0)
    colors = ["#e6550d" if v > 0 else "#2171b5" if v < 0 else "gray"
              for v in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", ls="-", lw=1)
    ax.axvline(d.mean(), color="red", ls="--", lw=1.5,
               label=f"Mean = {d.mean():+.2f}")
    ax.axvline(d.median(), color="darkred", ls=":", lw=1.5,
               label=f"Median = {d.median():+.1f}")
    ax.set_xlabel("Score Difference (Gemini − GPT)")
    ax.set_ylabel("Count")
    ax.set_title("Paired Score Differences", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "G11_paired_diff.png")


def graph_g12(df):
    """G12: Rep consistency heatmap (mean CV)."""
    header("G12: Rep consistency heatmap")
    _n = len(MODEL_NAMES)
    _ncols = min(_n, 3)
    _nrows = math.ceil(_n / _ncols)
    fig, _axes_arr = plt.subplots(_nrows, _ncols, figsize=(7 * _ncols, 5 * _nrows), squeeze=False)
    _axes_flat = [_axes_arr[r][c] for r in range(_nrows) for c in range(_ncols)]
    for _unused_j in range(_n, _nrows * _ncols):
        _axes_flat[_unused_j].set_visible(False)
    for j, model in enumerate(MODEL_NAMES):
        sub = df[df["model"] == model]
        mat = np.zeros((4, 5))
        for ai, axis in enumerate(AXES_ORDER):
            for li, lv in enumerate(LEVELS):
                grp = sub[(sub["axis"] == axis) & (sub["level"] == lv)]
                cvs = []
                for _, g in grp.groupby("article"):
                    scores = g["score"].values
                    m = scores.mean()
                    if m > 0 and len(scores) == 3:
                        cvs.append(scores.std(ddof=1) / m)
                mat[ai, li] = np.mean(cvs) if cvs else 0

        ax = _axes_flat[j]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0,
                       vmax=max(0.5, mat.max()))
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{lv:.1f}" for lv in LEVELS])
        ax.set_yticks(range(4))
        ax.set_yticklabels([AXIS_LABELS[a] for a in AXES_ORDER])
        ax.set_xlabel("Degradation Level")
        ax.set_title(model, fontweight="bold")
        for ai in range(4):
            for li in range(5):
                ax.text(li, ai, f"{mat[ai, li]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if mat[ai, li] > mat.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean CV")

    fig.suptitle("Scoring Consistency (CV across 3 reps, lower = more consistent)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G12_rep_consistency.png")


def graph_g13(df):
    """G13: Category effects boxplot (undegraded only)."""
    header("G13: Category effects")
    und = df[df["level"] == 0.0].copy()
    # Sort categories by median score
    cat_order = und.groupby("category")["score"].median().sort_values().index.tolist()

    _n = len(MODEL_NAMES)
    _ncols = min(_n, 2)
    _nrows = math.ceil(_n / _ncols)
    fig, _axes_arr = plt.subplots(_nrows, _ncols, figsize=(14, 5 * _nrows), squeeze=False)
    _axes_flat = [_axes_arr[r][c] for r in range(_nrows) for c in range(_ncols)]
    for _unused_j in range(_n, _nrows * _ncols):
        _axes_flat[_unused_j].set_visible(False)
    for j, model in enumerate(MODEL_NAMES):
        ax = _axes_flat[j]
        sub = und[und["model"] == model]
        data = [sub[sub["category"] == c]["score"].values for c in cat_order]
        bp = ax.boxplot(data, tick_labels=cat_order, patch_artist=True, vert=True,
                        medianprops=dict(color="black"))
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.6)
        ax.set_ylabel("Score")
        ax.set_title(model, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xticklabels(cat_order, rotation=45, ha="right", fontsize=9)
    fig.suptitle("Undegraded Scores by Article Category", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G13_category_effects.png")


def graph_g14(df):
    """G14: Residuals vs fitted + Q-Q plot."""
    header("G14: Residuals vs fitted")
    _n = len(MODEL_NAMES)
    fig, axes = plt.subplots(_n, 2, figsize=(12, 5 * _n), squeeze=False)
    for j, model in enumerate(MODEL_NAMES):
        sub = df[df["model"] == model]
        r = sp_stats.linregress(sub["level"], sub["score"])
        fitted = r.intercept + r.slope * sub["level"].values
        residuals = sub["score"].values - fitted

        # Residuals vs fitted
        ax = axes[j, 0]
        ax.scatter(fitted, residuals, s=1, alpha=0.1, color=COLORS[model], rasterized=True)
        # LOWESS
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            lw = lowess(residuals, fitted, frac=0.3)
            ax.plot(lw[:, 0], lw[:, 1], "r-", lw=2, label="LOWESS")
            ax.legend()
        except ImportError:
            pass
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{model} — Residuals vs Fitted", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Q-Q plot
        ax = axes[j, 1]
        sp_stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{model} — Q-Q Plot", fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "G14_residuals.png")


# ═══════════════════════════════════════════════════════════════
# T20: LOGPROB ENTROPY VS DEGRADATION LEVEL
# ═══════════════════════════════════════════════════════════════

def test_t20(lp_df):
    header("T20: Logprob Entropy vs Degradation Level (local models)")
    if lp_df.empty:
        print("  [SKIP] No logprob data available")
        return
    res = {}
    for model in sorted(lp_df["model"].unique()):
        sub = lp_df[lp_df["model"] == model]
        print(f"\n  {model}:")
        level_entropies = {}
        for lv in LEVELS:
            lv_sub = sub[sub["level"] == lv]
            if lv_sub.empty:
                continue
            mean_h = float(lv_sub["entropy"].mean())
            std_h = float(lv_sub["entropy"].std())
            level_entropies[str(lv)] = {"mean": mean_h, "std": std_h}
            print(f"    Level {lv:.1f}: entropy = {mean_h:.3f} ± {std_h:.3f}")

        if len(sub) >= 2:
            tau, p = sp_stats.kendalltau(sub["level"], sub["entropy"])
            res[model] = {
                "level_entropies": level_entropies,
                "kendall_tau": float(tau),
                "p_value": float(p),
            }
            print(f"    Kendall's τ (level vs entropy): {tau:+.3f}  p={p:.2e}")
    RESULTS["T20"] = res


# ═══════════════════════════════════════════════════════════════
# T21: BOUNDARY PROBABILITY MASS P(0)+P(10)
# ═══════════════════════════════════════════════════════════════

def test_t21(lp_df):
    header("T21: Boundary Probability Mass P(0)+P(10) (local models)")
    if lp_df.empty:
        print("  [SKIP] No logprob data available")
        return
    res = {}
    for model in sorted(lp_df["model"].unique()):
        sub = lp_df[lp_df["model"] == model]
        print(f"\n  {model}:")

        mean_pb = float(sub["p_boundary"].mean())
        mean_p0 = float(sub["p_0"].mean())
        mean_p10 = float(sub["p_10"].mean())
        print(f"    Overall: P(0)={mean_p0:.4f}  P(10)={mean_p10:.4f}  "
              f"P(boundary)={mean_pb:.4f}")

        level_data = {}
        for lv in LEVELS:
            lv_sub = sub[sub["level"] == lv]
            if lv_sub.empty:
                continue
            level_data[str(lv)] = {
                "p_0": float(lv_sub["p_0"].mean()),
                "p_10": float(lv_sub["p_10"].mean()),
                "p_boundary": float(lv_sub["p_boundary"].mean()),
            }
            print(f"    Level {lv:.1f}: P(0)={lv_sub['p_0'].mean():.4f}  "
                  f"P(10)={lv_sub['p_10'].mean():.4f}  "
                  f"P(boundary)={lv_sub['p_boundary'].mean():.4f}")

        info = {
            "overall_p_boundary": mean_pb,
            "overall_p_0": mean_p0,
            "overall_p_10": mean_p10,
            "level_data": level_data,
        }
        max_deg = sub[sub["level"] == 0.8]
        no_deg = sub[sub["level"] == 0.0]
        if not max_deg.empty:
            info["p0_at_max_degradation"] = float(max_deg["p_0"].mean())
            print(f"    ⇒ P(0) at max degradation (λ=0.8): {max_deg['p_0'].mean():.4f}")
        if not no_deg.empty:
            info["p10_at_no_degradation"] = float(no_deg["p_10"].mean())
            print(f"    ⇒ P(10) at no degradation (λ=0.0): {no_deg['p_10'].mean():.4f}")

        res[model] = info
    RESULTS["T21"] = res


# ═══════════════════════════════════════════════════════════════
# G15: COMPRESSION RATIO VS MODEL SIZE
# ═══════════════════════════════════════════════════════════════

def graph_g15(df):
    """G15: Compression ratio and sensitivity slope vs model size (local models)."""
    header("G15: Compression ratio vs model size")

    if not MODEL_SIZE_B:
        print("  [SKIP] No model_size_b entries in config")
        return

    ideal_range = 8.0
    points = []
    for model in MODEL_NAMES:
        if model not in MODEL_SIZE_B:
            continue
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        mean_0 = sub[sub["level"] == 0.0]["score"].mean()
        mean_08 = sub[sub["level"] == 0.8]["score"].mean()
        compression = (mean_0 - mean_08) / ideal_range
        slopes = []
        for axis in AXES_ORDER:
            ax_sub = sub[sub["axis"] == axis]
            if len(ax_sub) >= 2:
                r = sp_stats.linregress(ax_sub["level"], ax_sub["score"])
                slopes.append(r.slope)
        if not slopes:
            continue
        points.append({
            "model": model,
            "size_b": MODEL_SIZE_B[model],
            "compression": compression,
            "avg_slope": float(np.mean(slopes)),
        })

    if len(points) < 2:
        print("  [SKIP] Fewer than 2 local models scored — run scoring first")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title, refval, reflabel in [
        (axes[0], "compression", "Compression Ratio",
         "Compression Ratio vs Model Size", 1.0, "Perfect calibration (1.0)"),
        (axes[1], "avg_slope", "Average Sensitivity Slope (β₁)",
         "Sensitivity Slope vs Model Size", -10.0, "Perfect calibration (−10)"),
    ]:
        sizes = [p["size_b"] for p in points]
        vals  = [p[metric] for p in points]
        for p in points:
            ax.scatter(p["size_b"], p[metric], s=130,
                       color=COLORS.get(p["model"], "#555"), zorder=3)
            ax.annotate(p["model"], (p["size_b"], p[metric]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        # Trend line when enough points
        if len(points) >= 3:
            z = np.polyfit(sizes, vals, 1)
            xs = np.linspace(min(sizes), max(sizes), 100)
            ax.plot(xs, np.poly1d(z)(xs), "--", color="gray", alpha=0.6, lw=1.5)
        ax.axhline(refval, ls=":", color="gray", alpha=0.5, label=reflabel)
        ax.set_xlabel("Model Size (B parameters)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Compression Effect vs Model Scale (local models)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G15_compression_vs_size.png")


# ═══════════════════════════════════════════════════════════════
# G16: SCORE PROBABILITY HEATMAP
# ═══════════════════════════════════════════════════════════════

def graph_g16(lp_df):
    """G16: Score probability heatmap by degradation level (per local model)."""
    header("G16: Score probability heatmap")
    if lp_df.empty:
        print("  [SKIP] No logprob data available")
        return

    models = sorted(lp_df["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5),
                             squeeze=False)

    for j, model in enumerate(models):
        ax = axes[0, j]
        sub = lp_df[lp_df["model"] == model]
        mat = np.zeros((len(LEVELS), 11))
        for li, lv in enumerate(LEVELS):
            lv_sub = sub[sub["level"] == lv]
            if lv_sub.empty:
                continue
            for si in range(11):
                mat[li, si] = lv_sub[f"p_{si}"].mean()

        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(11))
        ax.set_xticklabels(range(11))
        ax.set_yticks(range(len(LEVELS)))
        ax.set_yticklabels([f"{lv:.1f}" for lv in LEVELS])
        ax.set_xlabel("Score")
        ax.set_ylabel("Degradation Level")
        ax.set_title(model, fontweight="bold")

        for li in range(len(LEVELS)):
            for si in range(11):
                val = mat[li, si]
                color = "white" if val > mat.max() * 0.5 else "black"
                ax.text(si, li, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean P(score)")

    fig.suptitle("Score Probability Distribution by Degradation Level",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G16_score_prob_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# G17: ENTROPY VS DEGRADATION LEVEL
# ═══════════════════════════════════════════════════════════════

def graph_g17(lp_df):
    """G17: Entropy of score distribution vs degradation level."""
    header("G17: Entropy vs degradation level")
    if lp_df.empty:
        print("  [SKIP] No logprob data available")
        return

    models = sorted(lp_df["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models:
        sub = lp_df[lp_df["model"] == model]
        color = COLORS.get(model, "#888888")
        means, lo, hi = [], [], []
        for lv in LEVELS:
            lv_sub = sub[sub["level"] == lv]["entropy"]
            if lv_sub.empty:
                means.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            m = lv_sub.mean()
            se = lv_sub.std() / np.sqrt(len(lv_sub))
            means.append(m)
            lo.append(m - 1.96 * se)
            hi.append(m + 1.96 * se)
        ax.plot(LEVELS, means, "o-", label=model, color=color, lw=2, ms=6)
        ax.fill_between(LEVELS, lo, hi, alpha=0.15, color=color)

    max_entropy = math.log2(11)
    ax.axhline(max_entropy, ls=":", color="gray", alpha=0.5,
               label=f"Max entropy ({max_entropy:.2f} bits)")

    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Score Distribution Entropy vs Degradation Level",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "G17_entropy_vs_level.png")


# ═══════════════════════════════════════════════════════════════
# G18: BOUNDARY MASS VS MODEL SIZE
# ═══════════════════════════════════════════════════════════════

def graph_g18(lp_df):
    """G18: Boundary probability mass P(0)+P(10) vs model size."""
    header("G18: Boundary mass vs model size")
    if lp_df.empty or not MODEL_SIZE_B:
        print("  [SKIP] No logprob data or model size metadata")
        return

    models = sorted(lp_df["model"].unique())
    models_with_size = [m for m in models if m in MODEL_SIZE_B]
    if len(models_with_size) < 2:
        print("  [SKIP] Fewer than 2 local models with size metadata")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: P(0) at max degradation vs model size
    ax = axes[0]
    for model in models_with_size:
        sub = lp_df[(lp_df["model"] == model) & (lp_df["level"] == 0.8)]
        if sub.empty:
            continue
        size = MODEL_SIZE_B[model]
        p0 = sub["p_0"].mean()
        ax.scatter(size, p0, s=130, color=COLORS.get(model, "#555"), zorder=3)
        ax.annotate(model, (size, p0), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Model Size (B parameters)")
    ax.set_ylabel("Mean P(score=0)")
    ax.set_title("P(0) at Max Degradation (λ=0.8)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2: P(10) at no degradation vs model size
    ax = axes[1]
    for model in models_with_size:
        sub = lp_df[(lp_df["model"] == model) & (lp_df["level"] == 0.0)]
        if sub.empty:
            continue
        size = MODEL_SIZE_B[model]
        p10 = sub["p_10"].mean()
        ax.scatter(size, p10, s=130, color=COLORS.get(model, "#555"), zorder=3)
        ax.annotate(model, (size, p10), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Model Size (B parameters)")
    ax.set_ylabel("Mean P(score=10)")
    ax.set_title("P(10) at No Degradation (λ=0.0)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Does the Model 'Know' the Extreme Scores? "
                 "Boundary Probability vs Scale",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "G18_boundary_mass_vs_size.png")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print(" No One Gets an A — Comprehensive Analysis")
    print(" 21 tests + 18 graphs")
    print("=" * 64)

    # ── Populate MODEL_NAMES, COLORS, MODEL_SIZE_B from config + data ──
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        for mc in cfg.get("llm_scoring", {}).get("models", []):
            if "model_size_b" in mc:
                MODEL_SIZE_B[mc["name"]] = mc["model_size_b"]

    # Load data
    print("\nLoading data...")
    df, paired, lp_df = load_data()

    discovered = sorted(df["model"].unique().tolist())
    MODEL_NAMES.extend(discovered)
    _n_extra = sum(1 for n in MODEL_NAMES if n not in _FIXED_COLORS)
    _tab20 = plt.cm.tab20(np.linspace(0, 1, max(_n_extra, 1)))
    _extra_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                     for r, g, b, _ in _tab20]
    _ei = 0
    for name in MODEL_NAMES:
        if name in _FIXED_COLORS:
            COLORS[name] = _FIXED_COLORS[name]
        else:
            COLORS[name] = _extra_colors[_ei]
            _ei += 1

    print(f"  Unified DataFrame: {len(df)} rows")
    print(f"  Paired DataFrame:  {len(paired)} rows")
    print(f"  Logprob DataFrame: {len(lp_df)} rows")
    print(f"  Models: {MODEL_NAMES}")
    print(f"  Articles: {df['article'].nunique()}")
    print(f"  Categories: {df['category'].nunique()}")

    # ── PRIMARY OUTCOMES ──
    header("PRIMARY OUTCOMES (declared pre-analysis)")
    print("  1. Compression ratio per model")
    print("  2. Average linear slope per model")
    print("  3. Median paired difference (Gemini − GPT)")

    # ── STATISTICAL TESTS ──
    test_t1(df)
    test_t2(df)
    test_t3(df)
    test_t4(df)
    t5_res = test_t5(df)
    test_t6(df)
    test_t7(df)
    test_t8(df)
    test_t9(df)
    test_t10(df)
    if not paired.empty:
        test_t11(paired)
        test_t12(paired)
        test_t13(paired)
    test_t14(df)
    test_t15(df)
    test_t16(df)
    if not paired.empty:
        test_t17(df, paired)
    test_t18()
    if not paired.empty:
        test_t19(paired)
    test_t20(lp_df)
    test_t21(lp_df)

    # ── PRIMARY OUTCOME SUMMARY ──
    header("PRIMARY OUTCOME SUMMARY")
    if "T3" in RESULTS:
        for m in MODEL_NAMES:
            r = RESULTS["T3"][m]
            print(f"  [{m}] Compression ratio: {r['compression_ratio']:.3f}")
    if "T5" in RESULTS:
        for m in MODEL_NAMES:
            slopes = [RESULTS["T5"][m][a]["slope"] for a in AXES_ORDER]
            print(f"  [{m}] Average slope: {np.mean(slopes):+.3f}")
    if "T12" in RESULTS:
        print(f"  Median paired diff (Gemini − GPT): "
              f"{RESULTS['T12']['median_diff']:+.2f}")

    # ── GRAPHS ──
    print(f"\n{'═' * 64}\n GENERATING GRAPHS\n{'═' * 64}")
    graph_g1(df)
    graph_g1b(df)
    graph_g2(df)
    graph_g3(df)
    graph_g4(df)
    graph_g5(df)
    graph_g6(df)
    graph_g7(df)
    graph_g8(df)
    graph_g9(df)
    if not paired.empty:
        graph_g10(paired)
        graph_g11(paired)
    graph_g12(df)
    graph_g13(df)
    graph_g14(df)
    graph_g15(df)
    graph_g16(lp_df)
    graph_g17(lp_df)
    graph_g18(lp_df)

    # ── SAVE RESULTS ──
    results_path = ANALYSIS_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {results_path.relative_to(ROOT)}")

    # Save summary table
    summary_rows = []
    if "T5" in RESULTS:
        for m in MODEL_NAMES:
            for a in AXES_ORDER:
                r = RESULTS["T5"][m][a]
                summary_rows.append({
                    "model": m, "axis": a, "slope": r["slope"],
                    "intercept": r["intercept"], "r_squared": r["r_squared"],
                    "p_value": r["p_value"]
                })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(ANALYSIS_DIR / "regression_summary.csv",
                                           index=False)
        print(f"  Summary saved to output/analysis/regression_summary.csv")

    print(f"\n{'═' * 64}")
    print(f" DONE — {len(RESULTS)} tests completed, 18 figures saved")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
