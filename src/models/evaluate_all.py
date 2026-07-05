"""
evaluate_all.py — The one true evaluation table.
Evaluates ALL trained models on the SAME held-out TEST split.
Writes:
  - reports/evaluation_results.csv  (authoritative test metrics, split=test)
  - reports/model_comparison.json   (ranked list for frontend)
  - reports/figures/pr_<model>.png, roc_<model>.png, confusion_<model>.png
  - reports/figures/model_comparison_bar.png
  - Registers every model in models/model_registry.json
  - Appends one record per model to reports/experiments.json

RUN: python -m src.models.evaluate_all
"""

import os
import json
import time
import hashlib
import datetime
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_curve,
)

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.registry.model_registry import register, set_production
from src.experiments.experiment_tracker import log_experiment
from src.models.train_river_online import RiverHSTWrapper

logger = get_logger("evaluate_all")
settings = get_settings()

os.makedirs("reports/figures", exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _feature_hash(feature_cols: list) -> str:
    """MD5 of sorted feature names — stable identifier for the feature contract."""
    return hashlib.md5(json.dumps(sorted(feature_cols)).encode()).hexdigest()[:8]


def _latency_minutes(test_clean: pd.DataFrame, preds: np.ndarray) -> float:
    is_anomaly = test_clean["label"].values
    in_window  = False
    window_starts = []
    for i in range(len(is_anomaly)):
        if is_anomaly[i] == 1 and not in_window:
            window_starts.append(i)
            in_window = True
        elif is_anomaly[i] == 0:
            in_window = False

    latencies = []
    for start in window_starts:
        detected = False
        for step in range(start, len(is_anomaly)):
            if is_anomaly[step] == 0:
                break
            if preds[step] == 1:
                latencies.append((step - start) * 5.0)  # 5 minutes per step
                detected = True
                break
        if not detected:
            window_length = sum(1 for s in range(start, len(is_anomaly)) if is_anomaly[s] == 1)
            latencies.append(window_length * 5.0)
    return float(np.mean(latencies)) if latencies else 0.0

def _window_metrics(y_test: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
    """Computes NAB-style windowed metrics: Precision, Recall, F1."""
    # Find true anomaly windows
    in_window = False
    windows = []
    current_window = []
    for i, val in enumerate(y_test):
        if val == 1:
            current_window.append(i)
            in_window = True
        elif in_window:
            windows.append(current_window)
            current_window = []
            in_window = False
    if in_window and current_window:
        windows.append(current_window)
        
    if not windows:
        return 1.0, 1.0, 1.0
        
    detected_windows = sum(1 for w in windows if any(preds[i] == 1 for i in w))
    window_recall = float(detected_windows / len(windows))
    
    # Find predicted segments
    in_pred = False
    pred_segments = []
    current_pred = []
    for i, p in enumerate(preds):
        if p == 1:
            current_pred.append(i)
            in_pred = True
        elif in_pred:
            pred_segments.append(current_pred)
            current_pred = []
            in_pred = False
    if in_pred and current_pred:
        pred_segments.append(current_pred)
        
    if not pred_segments:
        return 0.0, window_recall, 0.0
        
    # A predicted segment is a TP if it overlaps with ANY true window
    true_anomaly_indices = set(np.where(y_test == 1)[0])
    tp_segments = sum(1 for seg in pred_segments if any(idx in true_anomaly_indices for idx in seg))
    
    window_precision = float(tp_segments / len(pred_segments))
    
    window_f1 = 0.0
    if window_precision + window_recall > 0:
        window_f1 = 2 * (window_precision * window_recall) / (window_precision + window_recall)
        
    return window_precision, window_recall, window_f1


def _score_model(name: str, scores: np.ndarray, preds: np.ndarray,
                 y_test: np.ndarray, test_clean: pd.DataFrame,
                 inf_ms: float) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0.0  # type: ignore[call-overload]
    )
    try:
        roc_auc = roc_auc_score(y_test, scores)
        pr_p, pr_r, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(pr_r, pr_p)
    except Exception:
        roc_auc, pr_auc = 0.5, 0.0

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-9)
    latency = _latency_minutes(test_clean, preds)
    win_prec, win_rec, win_f1 = _window_metrics(y_test, preds)

    throughput = 1000.0 / (inf_ms + 1e-9)

    result = {
        "Model":                    name,
        "split":                    "test",
        "Precision (Point)":        float(prec),
        "Recall (Point)":           float(rec),
        "F1":                       float(f1),
        "Precision (Window)":       float(win_prec),
        "Recall (Window)":          float(win_rec),
        "F1 (Window)":              float(win_f1),
        "ROC-AUC":                  float(roc_auc),
        "PR-AUC":                   float(pr_auc),
        "False Alarm Rate":         float(far),
        "TP":                       int(tp),
        "TN":                       int(tn),
        "FP":                       int(fp),
        "FN":                       int(fn),
        "Detection Latency (min)":  float(latency),
        "Avg Inference Time (ms)":  float(inf_ms),
        "Throughput (readings/s)":  float(throughput),
    }
    logger.info(
        f"[{name}] WinRec={win_rec:.4f} WinPrec={win_prec:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} "
        f"PR-AUC={pr_auc:.4f} FAR={far:.4f} "
        f"lat={latency:.1f}min inf={inf_ms:.4f}ms"
    )
    return result, cm, scores  # type: ignore[return-value]


def _save_figures(name: str, scores: np.ndarray, preds: np.ndarray,
                  y_test: np.ndarray, cm: np.ndarray):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

        # PR curve
        pr_p, pr_r, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(pr_r, pr_p)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(pr_r, pr_p, lw=1.5)
        ax.set_title(f"PR Curve — {name} (AUC={pr_auc:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(f"reports/figures/pr_{slug}.png", dpi=100)
        plt.close(fig)

        # ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_auc = roc_auc_score(y_test, scores)
        except Exception:
            fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=1.5, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"ROC Curve — {name}")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"reports/figures/roc_{slug}.png", dpi=100)
        plt.close(fig)

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"], ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        fig.tight_layout()
        fig.savefig(f"reports/figures/confusion_{slug}.png", dpi=100)
        plt.close(fig)

    except Exception as e:
        logger.warning(f"Figure generation failed for {name}: {e}")


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_all():
    logger.info("=== evaluate_all: starting unified model evaluation ===")

    if not os.path.exists(settings.PROCESSED_CSV):
        raise FileNotFoundError(f"Processed CSV not found at {settings.PROCESSED_CSV}")

    df = pd.read_csv(settings.PROCESSED_CSV)

    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
    feat_hash = _feature_hash(feature_cols)

    test_clean = df[df["split"] == "test"].dropna(subset=feature_cols).copy()  # type: ignore[call-overload]
    X_test = test_clean[feature_cols].values  # numpy for fast scoring
    y_test = test_clean["label"].values

    logger.info(f"Test set: {len(test_clean)} rows | anomaly rate: {float(y_test.mean())*100:.2f}%")  # type: ignore[union-attr]

    results   = []
    all_scores = {}  # name -> scores (for comparison chart)

    # ── 1. Rolling Z-score baseline ──────────────────────────────────────────
    bpath = os.path.join(settings.MODEL_DIR, "threshold_zscore.json")
    if os.path.exists(bpath):
        with open(bpath, "r") as f:
            bcfg = json.load(f)
        k = bcfg["k"]
        zscores = np.abs(test_clean["zscore_15"].values)

        t0 = time.perf_counter()
        for _ in range(20): _ = zscores > k
        inf_ms = (time.perf_counter() - t0) / 20 / len(zscores) * 1000

        preds = (zscores > k).astype(int)
        row, cm, sc = _score_model("Rolling Z-score (Baseline)", zscores, preds, y_test, test_clean, inf_ms)
        results.append(row)
        all_scores["Rolling Z-score (Baseline)"] = zscores
        _save_figures("Rolling Z-score (Baseline)", zscores, preds, y_test, cm)

        register({
            "name": "rolling_zscore",
            "type": "baseline",
            "artifact_path": None,
            "threshold_path": bpath,
            "threshold": float(k),
            "feature_set_hash": feat_hash,
            "test_metrics": {k2: v for k2, v in row.items() if k2 not in ("Model", "split")},
            "is_production": False,
            "notes": "Rolling Z-score baseline on zscore_15 feature",
        })
        log_experiment({"model": "rolling_zscore", "split": "test", "feature_set_hash": feat_hash,
                        "threshold": float(k), **{k2: v for k2, v in row.items() if k2 not in ("Model", "split")}})
    else:
        logger.warning("Baseline threshold not found — skipping Rolling Z-score")

    # ── 2. Isolation Forest ───────────────────────────────────────────────────
    if_path = os.path.join(settings.MODEL_DIR, "isolation_forest.pkl")
    if os.path.exists(if_path):
        model = joblib.load(if_path)
        with open(os.path.join(settings.MODEL_DIR, "threshold.json")) as f:
            thresh = json.load(f)["threshold"]

        # Pass as DataFrame so sklearn sees feature names (no UserWarning)
        X_df = pd.DataFrame(X_test, columns=feature_cols)
        t0 = time.perf_counter()
        if_scores = -model.score_samples(X_df)
        inf_ms = (time.perf_counter() - t0) / len(X_test) * 1000

        preds = (if_scores > thresh).astype(int)
        row, cm, _ = _score_model("Isolation Forest", if_scores, preds, y_test, test_clean, inf_ms)
        results.append(row)
        all_scores["Isolation Forest"] = if_scores
        _save_figures("Isolation Forest", if_scores, preds, y_test, cm)

        register({
            "name": "isolation_forest",
            "type": "isolation_forest",
            "artifact_path": if_path,
            "threshold_path": os.path.join(settings.MODEL_DIR, "threshold.json"),
            "threshold": float(thresh),
            "feature_set_hash": feat_hash,
            "test_metrics": {k2: v for k2, v in row.items() if k2 not in ("Model", "split")},
            "is_production": True,
            "notes": "Primary production model. Fit on normal training rows.",
        })
        log_experiment({"model": "isolation_forest", "split": "test", "feature_set_hash": feat_hash,
                        "threshold": float(thresh), **{k2: v for k2, v in row.items() if k2 not in ("Model", "split")}})

    # ── 3. One-Class SVM ──────────────────────────────────────────────────────
    ocsvm_path = os.path.join(settings.MODEL_DIR, "one_class_svm.pkl")
    if os.path.exists(ocsvm_path):
        model = joblib.load(ocsvm_path)
        with open(os.path.join(settings.MODEL_DIR, "threshold_ocsvm.json")) as f:
            tcfg = json.load(f)
        thresh = tcfg["threshold"]

        t0 = time.perf_counter()
        ocsvm_scores = -model.decision_function(X_df)
        inf_ms = (time.perf_counter() - t0) / len(X_test) * 1000

        preds = (ocsvm_scores > thresh).astype(int)
        row, cm, _ = _score_model("One-Class SVM", ocsvm_scores, preds, y_test, test_clean, inf_ms)
        results.append(row)
        all_scores["One-Class SVM"] = ocsvm_scores
        _save_figures("One-Class SVM", ocsvm_scores, preds, y_test, cm)

        register({
            "name": "one_class_svm",
            "type": "one_class_svm",
            "artifact_path": ocsvm_path,
            "threshold_path": os.path.join(settings.MODEL_DIR, "threshold_ocsvm.json"),
            "threshold": float(thresh),
            "feature_set_hash": feat_hash,
            "test_metrics": {k2: v for k2, v in row.items() if k2 not in ("Model", "split")},
            "is_production": False,
            "notes": f"nu={tcfg.get('nu', 'N/A')}. RBF kernel, gamma=scale.",
        })
        log_experiment({"model": "one_class_svm", "split": "test", "feature_set_hash": feat_hash,
                        "threshold": float(thresh), **{k2: v for k2, v in row.items() if k2 not in ("Model", "split")}})

    # ── 4. LOF ────────────────────────────────────────────────────────────────
    lof_path = os.path.join(settings.MODEL_DIR, "lof.pkl")
    if os.path.exists(lof_path):
        model = joblib.load(lof_path)
        with open(os.path.join(settings.MODEL_DIR, "threshold_lof.json")) as f:
            tcfg = json.load(f)
        thresh = tcfg["threshold"]

        t0 = time.perf_counter()
        lof_scores = -model.score_samples(X_df)
        inf_ms = (time.perf_counter() - t0) / len(X_test) * 1000

        preds = (lof_scores > thresh).astype(int)
        row, cm, _ = _score_model("LOF", lof_scores, preds, y_test, test_clean, inf_ms)
        results.append(row)
        all_scores["LOF"] = lof_scores
        _save_figures("LOF", lof_scores, preds, y_test, cm)

        register({
            "name": "lof",
            "type": "lof",
            "artifact_path": lof_path,
            "threshold_path": os.path.join(settings.MODEL_DIR, "threshold_lof.json"),
            "threshold": float(thresh),
            "feature_set_hash": feat_hash,
            "test_metrics": {k2: v for k2, v in row.items() if k2 not in ("Model", "split")},
            "is_production": False,
            "notes": f"n_neighbors={tcfg.get('n_neighbors', 'N/A')}, novelty=True.",
        })
        log_experiment({"model": "lof", "split": "test", "feature_set_hash": feat_hash,
                        "threshold": float(thresh), **{k2: v for k2, v in row.items() if k2 not in ("Model", "split")}})

    # ── 5. Elliptic Envelope ──────────────────────────────────────────────────
    ee_path = os.path.join(settings.MODEL_DIR, "elliptic_envelope.pkl")
    if os.path.exists(ee_path):
        model = joblib.load(ee_path)
        with open(os.path.join(settings.MODEL_DIR, "threshold_elliptic.json")) as f:
            tcfg = json.load(f)
        thresh = tcfg["threshold"]

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ee_scores = -model.score_samples(X_df)
        inf_ms = (time.perf_counter() - t0) / len(X_test) * 1000

        preds = (ee_scores > thresh).astype(int)
        row, cm, _ = _score_model("Elliptic Envelope", ee_scores, preds, y_test, test_clean, inf_ms)
        results.append(row)
        all_scores["Elliptic Envelope"] = ee_scores
        _save_figures("Elliptic Envelope", ee_scores, preds, y_test, cm)

        register({
            "name": "elliptic_envelope",
            "type": "elliptic_envelope",
            "artifact_path": ee_path,
            "threshold_path": os.path.join(settings.MODEL_DIR, "threshold_elliptic.json"),
            "threshold": float(thresh),
            "feature_set_hash": feat_hash,
            "test_metrics": {k2: v for k2, v in row.items() if k2 not in ("Model", "split")},
            "is_production": False,
            "notes": tcfg.get("notes", ""),
        })
        log_experiment({"model": "elliptic_envelope", "split": "test", "feature_set_hash": feat_hash,
                        "threshold": float(thresh), **{k2: v for k2, v in row.items() if k2 not in ("Model", "split")}})

    # ── 6. LSTM Autoencoder ──────────────────────────────────────────────────
    lstm_path = os.path.join(settings.MODEL_DIR, "lstm_autoencoder.pkl")
    if os.path.exists(lstm_path):
        wrapper = joblib.load(lstm_path)
        # Ensure it runs on CPU for evaluation to be fair or if GPU not available
        wrapper.device = "cpu"
        wrapper.model.to("cpu")
        
        with open(os.path.join(settings.MODEL_DIR, "threshold_lstm.json")) as f:
            tcfg = json.load(f)
        thresh = tcfg["threshold"]

        t0 = time.perf_counter()
        # Wrapper score_samples returns negative MSE, lower = more anomalous
        lstm_scores = wrapper.score_samples(X_test)
        inf_ms = (time.perf_counter() - t0) / len(X_test) * 1000

        # anomaly if score < threshold (since scores are negative MSE)
        preds = (lstm_scores < thresh).astype(int)
        row, cm, _ = _score_model("LSTM Autoencoder", lstm_scores, preds, y_test, test_clean, inf_ms)
        results.append(row)
        all_scores["LSTM Autoencoder"] = lstm_scores
        _save_figures("LSTM Autoencoder", lstm_scores, preds, y_test, cm)

        register({
            "name": "lstm_autoencoder",
            "type": "lstm_autoencoder",
            "artifact_path": lstm_path,
            "threshold_path": os.path.join(settings.MODEL_DIR, "threshold_lstm.json"),
            "threshold": float(thresh),
            "feature_set_hash": feat_hash,
            "test_metrics": {k2: v for k2, v in row.items() if k2 not in ("Model", "split")},
            "is_production": False,
            "notes": tcfg.get("notes", ""),
        })
        log_experiment({"model": "lstm_autoencoder", "split": "test", "feature_set_hash": feat_hash,
                        "threshold": float(thresh), **{k2: v for k2, v in row.items() if k2 not in ("Model", "split")}})

    # ── 7. River HalfSpaceTrees ──────────────────────────────────────────────
    river_path = os.path.join(settings.MODEL_DIR, "river_online.pkl")
    if os.path.exists(river_path):
        wrapper = joblib.load(river_path)
        with open(os.path.join(settings.MODEL_DIR, "threshold_river.json")) as f:
            tcfg = json.load(f)
        thresh = tcfg["threshold"]

        t0 = time.perf_counter()
        # Returns sklearn-compatible negative scores
        river_scores_neg = wrapper.score_samples(X_df)
        inf_ms = (time.perf_counter() - t0) / len(X_test) * 1000
        
        # River raw scores are positive (higher = anomaly), evaluate_all works with inverted scores
        # We need to invert it back or use threshold properly.
        # Wait, the threshold tuned in train_river_online was done on inverted scores.
        # So we just do (scores > thresh) if we keep them inverted?
        # Let's see: train_river_online inverted the scores so higher is anomalous. 
        # Actually in train_river_online: val_scores = -wrapper.score_samples(X_val_df)
        # So threshold in json is for POSITIVE scores.
        # Let's convert negative to positive here:
        river_scores = -river_scores_neg

        preds = (river_scores > thresh).astype(int)
        row, cm, _ = _score_model("River HST (Online)", river_scores, preds, y_test, test_clean, inf_ms)
        results.append(row)
        all_scores["River HST (Online)"] = river_scores
        _save_figures("River HST (Online)", river_scores, preds, y_test, cm)

        register({
            "name": "river_hst",
            "type": "river_hst",
            "artifact_path": river_path,
            "threshold_path": os.path.join(settings.MODEL_DIR, "threshold_river.json"),
            "threshold": float(thresh),
            "feature_set_hash": feat_hash,
            "test_metrics": {k2: v for k2, v in row.items() if k2 not in ("Model", "split")},
            "is_production": False,
            "notes": "Online HalfSpaceTrees model using river",
        })
        log_experiment({"model": "river_hst", "split": "test", "feature_set_hash": feat_hash,
                        "threshold": float(thresh), **{k2: v for k2, v in row.items() if k2 not in ("Model", "split")}})

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["PR-AUC", "Detection Latency (min)"], ascending=[False, True]).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1
    results_df.to_csv("reports/evaluation_results.csv", index=False)
    logger.info(f"Saved evaluation_results.csv ({len(results_df)} models, split=test, ranked by PR-AUC)")

    # ── Save model_comparison.json ────────────────────────────────────────────
    comparison = []
    for _, row in results_df.iterrows():
        slug = str(row["Model"]).lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        comparison.append({
            "rank":            int(row["rank"]),
            "name":            row["Model"],
            "split":           "test",
            "precision":       round(float(row["Precision (Point)"]), 4),
            "recall":          round(float(row["Recall (Point)"]), 4),
            "f1":              round(float(row["F1"]), 4),
            "window_precision":round(float(row.get("Precision (Window)", 0.0)), 4),
            "window_recall":   round(float(row["Recall (Window)"]), 4),
            "window_f1":       round(float(row.get("F1 (Window)", 0.0)), 4),
            "roc_auc":         round(float(row["ROC-AUC"]), 4),
            "pr_auc":          round(float(row["PR-AUC"]), 4),
            "false_alarm_rate": round(float(row["False Alarm Rate"]), 4),
            "tp":              int(row["TP"]),
            "tn":              int(row["TN"]),
            "fp":              int(row["FP"]),
            "fn":              int(row["FN"]),
            "detection_latency_steps": round(float(row["Detection Latency (min)"]), 2), # Frontend uses steps field but we pass minutes now
            "avg_inference_ms": round(float(row["Avg Inference Time (ms)"]), 4),
            "throughput_rps":  round(float(row["Throughput (readings/s)"]), 1),
            "figures": {
                "pr":        f"reports/figures/pr_{slug}.png",
                "roc":       f"reports/figures/roc_{slug}.png",
                "confusion": f"reports/figures/confusion_{slug}.png",
            },
        })
    with open("reports/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved reports/model_comparison.json")

    # ── Comparison bar chart ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["Precision (Point)", "Recall (Point)", "F1", "PR-AUC"]
        x      = np.arange(len(metrics))
        n      = len(results_df)
        width  = 0.8 / n

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]
        for i, (_, row) in enumerate(results_df.iterrows()):
            vals = [row[m] for m in metrics]
            ax.bar(x + i * width, vals, width, label=row["Model"], color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x + width * (n - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Precision, Recall, F1, PR-AUC (Test Set)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("reports/figures/model_comparison_bar.png", dpi=100)
        plt.close(fig)
        logger.info("Saved model_comparison_bar.png")
    except Exception as e:
        logger.warning(f"Comparison bar chart failed: {e}")

    # ── Print ranking ─────────────────────────────────────────────────────────
    logger.info("=== RANKING (by PR-AUC, test set) ===")
    for _, row in results_df.iterrows():
        logger.info(
            f"#{int(row['rank'])} {row['Model']:35s} "
            f"PR-AUC={row['PR-AUC']:.4f}  WinRec={row['Recall (Window)']:.4f}  Lat={row['Detection Latency (min)']:.1f}m"
        )
    logger.info("=== evaluate_all complete ===")


if __name__ == "__main__":
    evaluate_all()
