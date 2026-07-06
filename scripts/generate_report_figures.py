"""
generate_report_figures.py — Data-driven figures for docs/final_report/figures/.

Everything here is computed from real, already-committed artifacts:
  - reports/evaluation_results.csv   (test-split metrics for all 7 models)
  - data/processed/nab_processed.csv (the same test split evaluate_all.py uses)
  - models/*.pkl + models/threshold_*.json (the actual trained models/thresholds)

The PR/ROC overlay curves re-load each trained model and score the same test
split evaluate_all.py scores at evaluation time — the scoring code below is a
direct copy of the per-model logic in src/models/evaluate_all.py, since that
script computes these same arrays in memory but never persists them (it only
saves the final per-model PNGs and the summary CSV/JSON). No numbers here are
invented; a model is skipped (and reported as skipped) if its artifact is
missing or fails to load/score.

RUN: python scripts/generate_report_figures.py
"""

import os
import sys
import json
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # so relative paths (models/, reports/, data/) resolve like evaluate_all.py expects

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score

from src.utils.config import get_settings
from src.models.train_river_online import RiverHSTWrapper  # noqa: F401 -- required so joblib can
# resolve RiverHSTWrapper when unpickling models/river_online.pkl (its __module__ was recorded as
# __main__ at training time; importing it here makes the name available in this script's __main__).

settings = get_settings()
FIG_DIR = ROOT / "docs" / "final_report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

skipped = []
notes = []


# ─── Part 1: bar charts straight from reports/evaluation_results.csv ────────

def load_results() -> pd.DataFrame:
    path = ROOT / "reports" / "evaluation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.models.evaluate_all` first")
    df = pd.read_csv(path)
    return df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)


def make_model_comparison_bar(df: pd.DataFrame):
    metrics = ["Precision (Point)", "Recall (Point)", "F1", "PR-AUC"]
    x = np.arange(len(metrics))
    n = len(df)
    width = 0.8 / n
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["Model"], color=colors[i], alpha=0.9)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Precision, Recall, F1, PR-AUC (Test Split)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison_bar.png", dpi=110)
    plt.close(fig)


def make_false_alarm_rate_comparison(df: pd.DataFrame):
    d = df.sort_values("False Alarm Rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#16a34a" if v < 0.1 else "#d97706" if v < 0.3 else "#dc2626" for v in d["False Alarm Rate"]]
    ax.barh(d["Model"], d["False Alarm Rate"], color=colors)
    for i, v in enumerate(d["False Alarm Rate"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("False Alarm Rate (FP / (FP + TN))")
    ax.set_title("False Alarm Rate by Model — Test Split (lower is better)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "false_alarm_rate_comparison.png", dpi=110)
    plt.close(fig)


def make_inference_latency_comparison(df: pd.DataFrame):
    d = df.sort_values("Avg Inference Time (ms)")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d["Model"], d["Avg Inference Time (ms)"], color="#2563eb")
    ax.set_xscale("log")
    for i, v in enumerate(d["Avg Inference Time (ms)"]):
        ax.text(v * 1.15, i, f"{v:.4g} ms", va="center", fontsize=8)
    ax.set_xlabel("Avg Inference Time per reading (ms, log scale)")
    ax.set_title("Inference Latency by Model — Test Split (lower is better)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "inference_latency_comparison.png", dpi=110)
    plt.close(fig)


# ─── Part 2: real overlaid PR/ROC curves — reload models, re-score test split ─

def load_test_split():
    df = pd.read_csv(settings.PROCESSED_CSV)
    with open(os.path.join(settings.MODEL_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    test_clean = df[df["split"] == "test"].dropna(subset=feature_cols).copy()
    X_test = test_clean[feature_cols].values
    X_df = pd.DataFrame(X_test, columns=feature_cols)
    y_test = test_clean["label"].values
    return test_clean, X_test, X_df, y_test


def score_all_models(X_test, X_df, y_test):
    """Mirrors the scoring formulas in src/models/evaluate_all.py exactly
    (same models, same test split, same sign conventions) so scores are
    directly comparable to reports/evaluation_results.csv."""
    scores = {}

    # Rolling Z-score baseline
    try:
        bpath = os.path.join(settings.MODEL_DIR, "threshold_zscore.json")
        if os.path.exists(bpath):
            zscores = np.abs(pd.read_csv(settings.PROCESSED_CSV).loc[
                pd.read_csv(settings.PROCESSED_CSV)["split"] == "test", "zscore_15"
            ].dropna().values)
            if len(zscores) == len(y_test):
                scores["Rolling Z-score (Baseline)"] = zscores
            else:
                notes.append("Rolling Z-score: row-count mismatch after dropna, skipped for curve overlay")
        else:
            skipped.append(("Rolling Z-score (Baseline)", "threshold_zscore.json not found"))
    except Exception as e:
        skipped.append(("Rolling Z-score (Baseline)", str(e)))

    # Isolation Forest
    try:
        path = os.path.join(settings.MODEL_DIR, "isolation_forest.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            scores["Isolation Forest"] = -model.score_samples(X_df)
        else:
            skipped.append(("Isolation Forest", "isolation_forest.pkl not found"))
    except Exception as e:
        skipped.append(("Isolation Forest", str(e)))

    # One-Class SVM
    try:
        path = os.path.join(settings.MODEL_DIR, "one_class_svm.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            scores["One-Class SVM"] = -model.decision_function(X_df)
        else:
            skipped.append(("One-Class SVM", "one_class_svm.pkl not found"))
    except Exception as e:
        skipped.append(("One-Class SVM", str(e)))

    # LOF
    try:
        path = os.path.join(settings.MODEL_DIR, "lof.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            scores["LOF"] = -model.score_samples(X_df)
        else:
            skipped.append(("LOF", "lof.pkl not found"))
    except Exception as e:
        skipped.append(("LOF", str(e)))

    # Elliptic Envelope
    try:
        path = os.path.join(settings.MODEL_DIR, "elliptic_envelope.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores["Elliptic Envelope"] = -model.score_samples(X_df)
        else:
            skipped.append(("Elliptic Envelope", "elliptic_envelope.pkl not found"))
    except Exception as e:
        skipped.append(("Elliptic Envelope", str(e)))

    # LSTM Autoencoder
    try:
        path = os.path.join(settings.MODEL_DIR, "lstm_autoencoder.pkl")
        if os.path.exists(path):
            wrapper = joblib.load(path)
            wrapper.device = "cpu"
            wrapper.model.to("cpu")
            scores["LSTM Autoencoder"] = wrapper.score_samples(X_test)  # lower = more anomalous
        else:
            skipped.append(("LSTM Autoencoder", "lstm_autoencoder.pkl not found"))
    except Exception as e:
        skipped.append(("LSTM Autoencoder", str(e)))

    # River HST (Online)
    try:
        path = os.path.join(settings.MODEL_DIR, "river_online.pkl")
        if os.path.exists(path):
            wrapper = joblib.load(path)
            scores["River HST (Online)"] = -wrapper.score_samples(X_df)  # inverted, higher = anomalous
        else:
            skipped.append(("River HST (Online)", "river_online.pkl not found"))
    except Exception as e:
        skipped.append(("River HST (Online)", str(e)))

    return scores


def make_pr_roc_overlays(y_test, scores: dict):
    if not scores:
        notes.append("pr_curve_all_models.png / roc_curve_all_models.png skipped — no model scored successfully")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(scores)))

    # PR curves — LSTM's raw score is "lower = anomalous", so flip sign for sklearn's
    # convention (higher score => positive class) when computing the curve.
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, sc), color in zip(scores.items(), colors):
        s = -sc if name == "LSTM Autoencoder" else sc
        p, r, _ = precision_recall_curve(y_test, s)
        pr_auc = auc(r, p)
        ax.plot(r, p, lw=1.6, label=f"{name} (AUC={pr_auc:.3f})", color=color)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Curves — All Models (Test Split)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve_all_models.png", dpi=110)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, sc), color in zip(scores.items(), colors):
        s = -sc if name == "LSTM Autoencoder" else sc
        try:
            fpr, tpr, _ = roc_curve(y_test, s)
            roc_auc = roc_auc_score(y_test, s)
        except Exception:
            continue
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={roc_auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models (Test Split)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve_all_models.png", dpi=110)
    plt.close(fig)


def main():
    df = load_results()
    make_model_comparison_bar(df)
    make_false_alarm_rate_comparison(df)
    make_inference_latency_comparison(df)

    try:
        test_clean, X_test, X_df, y_test = load_test_split()
        scores = score_all_models(X_test, X_df, y_test)
        make_pr_roc_overlays(y_test, scores)
        scored_names = list(scores.keys())
    except Exception as e:
        scored_names = []
        notes.append(f"PR/ROC overlay generation failed entirely: {e}")

    print("=== generate_report_figures.py done ===")
    print("Scored for PR/ROC overlay:", scored_names)
    if skipped:
        print("Skipped (model-level):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    if notes:
        print("Notes:")
        for n in notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
