#!/usr/bin/env python3
"""
Auxiliary Regressor Mitigation
===============================
Trains post-hoc regressors on logprob features to predict proxy ground truth.

Feature sets:
  A1 — entropy, p_ceiling, expected_score, top2_gap
  A2 — A1 + kl_vs_uniform, expected_minus_argmax
  B  — full 11-probability vector  (p_0 … p_10)
  C  — A2 + B  (everything)

Models: Ridge, XGBoost  →  up to 8 fitted models per LLM.

Uses GroupKFold (group = article) to prevent text leakage.
SHAP values for best-performing model.

Outputs:
  output/mitigations/results/aux_regressor.csv
  output/mitigations/intermediate/aux_predictions.csv
  output/mitigations/figures/aux_shap_summary.png
  output/mitigations/figures/aux_cv_comparison.png

Usage:  python scripts/mitigate_aux_regressor.py
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
    RANDOM_STATE, proxy_ground_truth, compute_compression_ratio,
    pairwise_accuracy,
)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ── Paths ───────────────────────────────────────────────────────
INT_DIR = ROOT / "output" / "mitigations" / "intermediate"
RES_DIR = ROOT / "output" / "mitigations" / "results"
FIG_DIR = ROOT / "output" / "mitigations" / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)

# ── Feature sets ────────────────────────────────────────────────
FEAT_A1 = ["entropy", "p_ceiling", "expected_score", "top2_gap"]
FEAT_A2 = FEAT_A1 + ["kl_vs_uniform", "expected_minus_argmax"]
FEAT_B = [f"p_{i}" for i in range(11)]
FEAT_C = FEAT_A2 + FEAT_B

FEATURE_SETS = {"A1": FEAT_A1, "A2": FEAT_A2, "B": FEAT_B, "C": FEAT_C}


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def eval_scores(scores: np.ndarray, levels: np.ndarray) -> dict:
    target = proxy_ground_truth(levels)
    wd = float(sp_stats.wasserstein_distance(scores, target))
    cr = compute_compression_ratio(scores)
    rho, rho_p = sp_stats.spearmanr(scores, target)
    slope = sp_stats.linregress(levels, scores).slope
    pa = pairwise_accuracy(scores, target)
    rmse = float(np.sqrt(np.mean((scores - target) ** 2)))
    return {
        "wasserstein": round(wd, 4),
        "compression_ratio": round(cr, 4),
        "spearman_rho": round(float(rho), 4),
        "dose_response_slope": round(float(slope), 4),
        "pairwise_accuracy": round(pa, 4),
        "rmse": round(rmse, 4),
    }


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
    all_preds = []
    best_models = {}  # model_name -> (fset, reg_name, Pipeline)

    for model_name in models:
        sub = lp[lp["model"] == model_name].copy().reset_index(drop=True)
        levels = sub["level"].values
        raw_scores = sub["score"].values.astype(float)
        target = proxy_ground_truth(levels)
        groups = sub["article"].values

        # Raw baseline
        m_raw = eval_scores(raw_scores, levels)
        m_raw.update({"model": model_name, "method": "raw", "feature_set": "-"})
        all_results.append(m_raw)

        best_rmse = np.inf
        best_key = None

        for fset_name, feat_cols in FEATURE_SETS.items():
            # Check columns exist
            missing = [c for c in feat_cols if c not in sub.columns]
            if missing:
                continue

            X = sub[feat_cols].values.astype(float)
            if np.any(np.isnan(X)):
                X = np.nan_to_num(X, nan=0.0)

            gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            regressors = {
                "Ridge": Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
                ]),
                "XGBoost": xgb.XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbosity=0,
                ),
            }

            for reg_name, reg in regressors.items():
                oof_preds = np.full(len(sub), np.nan)

                for train_idx, val_idx in gkf.split(X, target, groups):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr = target[train_idx]
                    reg_clone = _clone_model(reg)
                    reg_clone.fit(X_tr, y_tr)
                    oof_preds[val_idx] = np.clip(reg_clone.predict(X_val), 0, 10)

                # Handle any remaining NaNs from fold splits
                valid = ~np.isnan(oof_preds)
                if valid.sum() < len(sub) * 0.5:
                    continue

                m = eval_scores(oof_preds[valid], levels[valid])
                m.update({
                    "model": model_name,
                    "method": f"aux_{reg_name}",
                    "feature_set": fset_name,
                })
                all_results.append(m)

                if m["rmse"] < best_rmse:
                    best_rmse = m["rmse"]
                    best_key = (fset_name, reg_name)

                # Store predictions
                for i in np.where(valid)[0]:
                    all_preds.append({
                        "sample_id": sub.iloc[i]["sample_id"],
                        "model": model_name,
                        "method": f"aux_{reg_name}_{fset_name}",
                        "predicted": round(oof_preds[i], 4),
                        "target": round(target[i], 4),
                    })

        if best_key:
            # Fit final model on all data for SHAP
            fset_name, reg_name = best_key
            feat_cols = FEATURE_SETS[fset_name]
            X_all = sub[feat_cols].values.astype(float)
            X_all = np.nan_to_num(X_all, nan=0.0)
            reg_final = _clone_model(
                regressors[reg_name] if reg_name in regressors else
                xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                 learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0)
            )
            # Re-create regressors dict since we consumed it
            if reg_name == "Ridge":
                reg_final = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
                ])
            else:
                reg_final = xgb.XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbosity=0,
                )
            reg_final.fit(X_all, target)
            best_models[model_name] = (fset_name, reg_name, reg_final, feat_cols, X_all)

        print(f"  {model_name:30s}  best={best_key}  RMSE={best_rmse:.3f}")

    # Save results
    cols = ["model", "method", "feature_set", "wasserstein", "compression_ratio",
            "spearman_rho", "dose_response_slope", "pairwise_accuracy", "rmse"]
    pd.DataFrame(all_results)[cols].to_csv(
        RES_DIR / "aux_regressor.csv", index=False)
    print(f"\n  Saved → {RES_DIR / 'aux_regressor.csv'}")

    if all_preds:
        pd.DataFrame(all_preds).to_csv(
            INT_DIR / "aux_predictions.csv", index=False)
        print(f"  Saved → {INT_DIR / 'aux_predictions.csv'}")

    # ── SHAP figure ─────────────────────────────────────────────
    if best_models:
        _plot_shap(best_models)

    # ── CV comparison bar chart ─────────────────────────────────
    _plot_cv_comparison(all_results)

    print("\nDone.")


def _clone_model(model):
    """Simple clone for sklearn/xgb models."""
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        # Fallback for xgboost
        params = model.get_params()
        return type(model)(**params)


def _plot_shap(best_models: dict):
    """SHAP summary plot for the best model of each LLM."""
    try:
        import shap
    except ImportError:
        print("  [SKIP] shap not installed, skipping SHAP figure")
        return

    n = len(best_models)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, (model_name, (fset, reg_name, reg, feat_cols, X_all)) in enumerate(
            best_models.items()):
        ax = axes[idx // ncols][idx % ncols]

        # Get the actual estimator for SHAP
        if reg_name == "Ridge":
            # For Ridge in pipeline, use linear explainer on transformed data
            scaler = reg.named_steps["scaler"]
            ridge = reg.named_steps["reg"]
            X_scaled = scaler.transform(X_all)
            explainer = shap.LinearExplainer(ridge, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        else:
            explainer = shap.TreeExplainer(reg)
            shap_values = explainer.shap_values(X_all)

        # Manual bar plot of mean |SHAP|
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_k = min(10, len(feat_cols))
        labels = [feat_cols[i] for i in order[:top_k]]
        values = mean_abs[order[:top_k]]

        ax.barh(range(top_k), values[::-1], color="#1f77b4")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_name}\n({reg_name}, {fset})", fontsize=10)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("SHAP Feature Importance (Best Auxiliary Regressor)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aux_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'aux_shap_summary.png'}")


def _plot_cv_comparison(all_results: list[dict]):
    """Bar chart comparing RMSE across feature sets and regressor types."""
    df = pd.DataFrame(all_results)
    df = df[df["method"] != "raw"].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    df["label"] = df["model"] + "\n" + df["feature_set"] + " " + df["method"]
    df_sorted = df.sort_values(["model", "rmse"])

    # Group by model, show rmse
    models = sorted(df["model"].unique())
    n_models = len(models)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_models, 1)))
    x_pos = 0
    ticks, tick_labels = [], []

    for m_idx, model in enumerate(models):
        msub = df_sorted[df_sorted["model"] == model]
        for _, row in msub.iterrows():
            ax.bar(x_pos, row["rmse"], color=colors[m_idx], edgecolor="white",
                   width=0.8)
            ticks.append(x_pos)
            tick_labels.append(f"{row['feature_set']}\n{row['method'].replace('aux_', '')}")
            x_pos += 1
        x_pos += 0.5  # gap between models

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("RMSE vs Proxy GT")
    ax.set_title("Auxiliary Regressor CV Performance", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aux_cv_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'aux_cv_comparison.png'}")


if __name__ == "__main__":
    main()
