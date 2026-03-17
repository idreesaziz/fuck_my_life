"""
Calibration Recovery Analysis
=============================
Fits affine and sigmoid calibration functions to LLM scores, evaluates on
held-out articles, and generates a figure showing raw vs calibrated dose-response
curves against the ideal calibration line.

Output:
  - output/figures/G15_calibration_recovery.png
  - output/analysis/calibration_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, spearmanr
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT / "data" / "degraded" / "degraded_samples.json"
GPT_SCORES_PATH = ROOT / "data" / "scores" / "gpt5_mini_scores.json"
GEMINI_SCORES_PATH = ROOT / "data" / "scores" / "llm_scores_gemini.json"
FIG_PATH = ROOT / "output" / "figures" / "G15_calibration_recovery.png"
RESULTS_PATH = ROOT / "output" / "analysis" / "calibration_results.json"

# ── Load data ──────────────────────────────────────────────────
samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
gpt_scores = json.loads(GPT_SCORES_PATH.read_text(encoding="utf-8"))
gemini_scores = json.loads(GEMINI_SCORES_PATH.read_text(encoding="utf-8"))

# Build arrays: for each sample, get title, axis, level, gpt_score, gemini_score
n = len(samples)
titles = [s["source_title"] for s in samples]
levels = np.array([s["level"] for s in samples])
axes = [s["axis"] for s in samples]
gpt = np.array([gpt_scores[i]["score"] for i in range(n)])
gem = np.array([gemini_scores[i]["score"] for i in range(n)])

# Ground truth: ideal score = 10 * (1 - level)
ideal = 10.0 * (1.0 - levels)

# ── Train/test split by article (80/20) ───────────────────────
unique_titles = sorted(set(titles))
rng = np.random.RandomState(42)
rng.shuffle(unique_titles)
split = int(0.8 * len(unique_titles))
train_titles = set(unique_titles[:split])
test_titles = set(unique_titles[split:])

train_mask = np.array([t in train_titles for t in titles])
test_mask = ~train_mask

print(f"Train articles: {len(train_titles)}, Test articles: {len(test_titles)}")
print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")


# ── Calibration functions ──────────────────────────────────────
def affine(x, a, b):
    return a * x + b


def sigmoid(x, a, b, c, d):
    """Generalized sigmoid: d + c / (1 + exp(-a * (x - b)))"""
    return d + c / (1.0 + np.exp(-a * (x - b)))


# ── Fit and evaluate ──────────────────────────────────────────
results = {}

for model_name, scores in [("GPT-5 mini", gpt), ("Gemini 3 Flash", gem)]:
    model_results = {}

    # Train data
    s_train = scores[train_mask].astype(float)
    ideal_train = ideal[train_mask]

    # Test data
    s_test = scores[test_mask].astype(float)
    ideal_test = ideal[test_mask]
    levels_test = levels[test_mask]

    # --- Raw (uncalibrated) metrics on test set ---
    tau_raw, _ = kendalltau(ideal_test, s_test)
    rho_raw, _ = spearmanr(ideal_test, s_test)
    rmse_raw = np.sqrt(np.mean((s_test - ideal_test) ** 2))

    # Compression ratio on test set
    mean_by_level_raw = {}
    for lv in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask_lv = levels_test == lv
        if mask_lv.any():
            mean_by_level_raw[lv] = s_test[mask_lv].mean()
    cr_raw = (mean_by_level_raw[0.0] - mean_by_level_raw[0.8]) / 8.0

    model_results["raw"] = {
        "kendall_tau": round(tau_raw, 4),
        "spearman_rho": round(rho_raw, 4),
        "rmse": round(rmse_raw, 4),
        "compression_ratio": round(cr_raw, 4),
    }

    # --- Affine calibration ---
    popt_aff, _ = curve_fit(affine, s_train, ideal_train)
    a_aff, b_aff = popt_aff
    s_test_aff = affine(s_test, a_aff, b_aff)

    tau_aff, _ = kendalltau(ideal_test, s_test_aff)
    rho_aff, _ = spearmanr(ideal_test, s_test_aff)
    rmse_aff = np.sqrt(np.mean((s_test_aff - ideal_test) ** 2))

    mean_by_level_aff = {}
    for lv in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask_lv = levels_test == lv
        if mask_lv.any():
            mean_by_level_aff[lv] = s_test_aff[mask_lv].mean()
    cr_aff = (mean_by_level_aff[0.0] - mean_by_level_aff[0.8]) / 8.0

    model_results["affine"] = {
        "params": {"a": round(a_aff, 4), "b": round(b_aff, 4)},
        "kendall_tau": round(tau_aff, 4),
        "spearman_rho": round(rho_aff, 4),
        "rmse": round(rmse_aff, 4),
        "compression_ratio": round(cr_aff, 4),
    }

    # --- Sigmoid calibration ---
    try:
        popt_sig, _ = curve_fit(
            sigmoid, s_train, ideal_train,
            p0=[1.0, 5.0, 10.0, 0.0],
            maxfev=10000,
        )
        s_test_sig = sigmoid(s_test, *popt_sig)
        tau_sig, _ = kendalltau(ideal_test, s_test_sig)
        rho_sig, _ = spearmanr(ideal_test, s_test_sig)
        rmse_sig = np.sqrt(np.mean((s_test_sig - ideal_test) ** 2))

        mean_by_level_sig = {}
        for lv in [0.0, 0.2, 0.4, 0.6, 0.8]:
            mask_lv = levels_test == lv
            if mask_lv.any():
                mean_by_level_sig[lv] = s_test_sig[mask_lv].mean()
        cr_sig = (mean_by_level_sig[0.0] - mean_by_level_sig[0.8]) / 8.0

        model_results["sigmoid"] = {
            "params": {k: round(v, 4) for k, v in zip("abcd", popt_sig)},
            "kendall_tau": round(tau_sig, 4),
            "spearman_rho": round(rho_sig, 4),
            "rmse": round(rmse_sig, 4),
            "compression_ratio": round(cr_sig, 4),
        }
    except RuntimeError:
        print(f"  Sigmoid fit failed for {model_name}")
        popt_sig = None
        model_results["sigmoid"] = {"error": "convergence failure"}

    results[model_name] = model_results

    print(f"\n{model_name}:")
    print(f"  Affine: a={a_aff:.4f}, b={b_aff:.4f}")
    print(f"  Raw  -> tau={tau_raw:.4f}, rho={rho_raw:.4f}, RMSE={rmse_raw:.4f}, CR={cr_raw:.4f}")
    print(f"  Affine -> tau={tau_aff:.4f}, rho={rho_aff:.4f}, RMSE={rmse_aff:.4f}, CR={cr_aff:.4f}")
    if popt_sig is not None:
        print(f"  Sigmoid -> tau={tau_sig:.4f}, rho={rho_sig:.4f}, RMSE={rmse_sig:.4f}, CR={cr_sig:.4f}")


# ── Generate figure ───────────────────────────────────────────
fig, axes_arr = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

level_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
ideal_curve = 10.0 * (1.0 - level_vals)

for ax, (model_name, scores_arr) in zip(axes_arr, [("GPT-5 mini", gpt), ("Gemini 3 Flash", gem)]):
    mr = results[model_name]

    # Compute mean scores per level on TEST set
    raw_means = []
    aff_means = []
    sig_means = []

    s_test_local = scores_arr[test_mask].astype(float)
    levels_test_local = levels[test_mask]

    # Affine params
    a_aff = mr["affine"]["params"]["a"]
    b_aff = mr["affine"]["params"]["b"]
    s_aff_local = affine(s_test_local, a_aff, b_aff)

    # Sigmoid params
    if "params" in mr.get("sigmoid", {}):
        sp = mr["sigmoid"]["params"]
        s_sig_local = sigmoid(s_test_local, sp["a"], sp["b"], sp["c"], sp["d"])
    else:
        s_sig_local = None

    for lv in level_vals:
        mask_lv = levels_test_local == lv
        raw_means.append(s_test_local[mask_lv].mean())
        aff_means.append(s_aff_local[mask_lv].mean())
        if s_sig_local is not None:
            sig_means.append(s_sig_local[mask_lv].mean())

    # Plot
    ax.plot(level_vals, ideal_curve, "k--", linewidth=2, label="Ideal ($S = 10(1-\\lambda)$)", zorder=5)
    ax.plot(level_vals, raw_means, "o-", color="#d62728", linewidth=2, markersize=8, label="Raw scores", zorder=4)
    ax.plot(level_vals, aff_means, "s-", color="#2ca02c", linewidth=2, markersize=8, label="Affine calibration", zorder=3)
    if sig_means:
        ax.plot(level_vals, sig_means, "^-", color="#1f77b4", linewidth=2, markersize=8, label="Sigmoid calibration", zorder=2)

    # Shade compression gap (raw)
    ax.fill_between(level_vals, raw_means, ideal_curve, alpha=0.12, color="#d62728")

    ax.set_xlabel("Degradation level ($\\lambda$)", fontsize=13)
    ax.set_title(model_name, fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(0, 11)
    ax.set_xticks(level_vals)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)

axes_arr[0].set_ylabel("Score", fontsize=13)

fig.suptitle("Calibration Recovery: Raw vs. Calibrated Dose-Response Curves (Held-Out Articles)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {FIG_PATH}")

# ── Save results ──────────────────────────────────────────────
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Results saved to {RESULTS_PATH}")
