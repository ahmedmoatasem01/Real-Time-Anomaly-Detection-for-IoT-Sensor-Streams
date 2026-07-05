import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("evaluate")
settings = get_settings()

def get_anomaly_windows(df: pd.DataFrame):
    df = df.copy()
    df["is_anomaly_diff"] = df["label"].diff()
    onsets = df[df["is_anomaly_diff"] == 1].index
    offsets = df[df["is_anomaly_diff"] == -1].index
    
    if df["label"].iloc[0] == 1:
        onsets = onsets.insert(0, df.index[0])
    if df["label"].iloc[-1] == 1:
        offsets = offsets.append(pd.Index([df.index[-1]]))
        
    windows = list(zip(onsets, offsets))
    return windows

def calculate_detection_latency(df: pd.DataFrame, preds: np.ndarray) -> float:
    df = df.reset_index(drop=True)
    is_anomaly = df["label"].values
    
    in_window = False
    window_starts = []
    for i in range(len(is_anomaly)):
        if is_anomaly[i] == 1 and not in_window:
            window_starts.append(i)
            in_window = True
        elif is_anomaly[i] == 0 and in_window:
            in_window = False
            
    latencies = []
    for start in window_starts:
        detected = False
        for step in range(start, len(is_anomaly)):
            if is_anomaly[step] == 0:
                break
            if preds[step] == 1:
                latencies.append(step - start)
                detected = True
                break
        if not detected:
            window_len = 0
            for step in range(start, len(is_anomaly)):
                if is_anomaly[step] == 0:
                    break
                window_len += 1
            latencies.append(window_len)
            
    if not latencies:
        return 0.0
    return float(np.mean(latencies))

def evaluate_models():
    logger.info("Starting model evaluation...")
    if not os.path.exists(settings.PROCESSED_CSV):
        raise FileNotFoundError(f"Processed CSV not found at {settings.PROCESSED_CSV}")
        
    df = pd.read_csv(settings.PROCESSED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
        
    test_df = df[df["split"] == "test"].copy()
    test_clean = test_df.dropna(subset=feature_cols)  # type: ignore[call-overload]
    
    X_test = test_clean[feature_cols].values
    y_test = test_clean["label"].values
    
    results = []
    
    # 1. EVALUATE BASELINE (ROLLING Z-SCORE)
    baseline_thresh_path = os.path.join(settings.MODEL_DIR, "threshold_zscore.json")
    with open(baseline_thresh_path, "r") as f:
        baseline_cfg = json.load(f)
    k = baseline_cfg["k"]
    
    zscores_test = np.abs(test_clean["zscore_15"].values)
    baseline_preds = (zscores_test > k).astype(int)
    
    b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(y_test, baseline_preds, average="binary", zero_division=0.0)  # type: ignore[call-overload]
    
    try:
        b_roc_auc = roc_auc_score(y_test, zscores_test)
        b_pr_precision, b_pr_recall, _ = precision_recall_curve(y_test, zscores_test)
        b_pr_auc = auc(b_pr_recall, b_pr_precision)
    except Exception:
        b_roc_auc = 0.5
        b_pr_auc = 0.0
        b_pr_precision = None  # type: ignore[assignment]
        b_pr_recall = None  # type: ignore[assignment]
        
    b_conf = confusion_matrix(y_test, baseline_preds)
    tn, fp, fn, tp = b_conf.ravel()
    b_far = fp / (fp + tn + 1e-9)
    b_latency = calculate_detection_latency(test_clean, baseline_preds)
    
    start_t = time.perf_counter()
    for _ in range(100):
        _ = zscores_test > k
    b_inf_time = (time.perf_counter() - start_t) / 100 / len(zscores_test) * 1000
    
    results.append({
        "Model": "Rolling Z-score (Baseline)",
        "Precision": b_prec,
        "Recall": b_rec,
        "F1": b_f1,
        "ROC-AUC": b_roc_auc,
        "PR-AUC": b_pr_auc,
        "False Alarm Rate": b_far,
        "Detection Latency (steps)": b_latency,
        "Avg Inference Time (ms)": b_inf_time
    })
    
    # 2. EVALUATE ISOLATION FOREST
    if_model_path = os.path.join(settings.MODEL_DIR, "isolation_forest.pkl")
    if_model = joblib.load(if_model_path)
    
    if_thresh_path = os.path.join(settings.MODEL_DIR, "threshold.json")
    with open(if_thresh_path, "r") as f:
        if_cfg = json.load(f)
    thresh = if_cfg["threshold"]
    
    start_t = time.perf_counter()
    if_scores = -if_model.score_samples(X_test)
    end_t = time.perf_counter()
    if_inf_time = (end_t - start_t) / len(X_test) * 1000
    
    if_preds = (if_scores > thresh).astype(int)
    
    if_prec, if_rec, if_f1, _ = precision_recall_fscore_support(y_test, if_preds, average="binary", zero_division=0.0)  # type: ignore[call-overload]
    
    try:
        if_roc_auc = roc_auc_score(y_test, if_scores)
        if_pr_precision, if_pr_recall, _ = precision_recall_curve(y_test, if_scores)
        if_pr_auc = auc(if_pr_recall, if_pr_precision)
    except Exception:
        if_roc_auc = 0.5
        if_pr_auc = 0.0
        if_pr_precision = None  # type: ignore[assignment]
        if_pr_recall = None  # type: ignore[assignment]
        
    if_conf = confusion_matrix(y_test, if_preds)
    tn_if, fp_if, fn_if, tp_if = if_conf.ravel()
    if_far = fp_if / (fp_if + tn_if + 1e-9)
    if_latency = calculate_detection_latency(test_clean, if_preds)
    
    results.append({
        "Model": "Isolation Forest (MVP)",
        "Precision": if_prec,
        "Recall": if_rec,
        "F1": if_f1,
        "ROC-AUC": if_roc_auc,
        "PR-AUC": if_pr_auc,
        "False Alarm Rate": if_far,
        "Detection Latency (steps)": if_latency,
        "Avg Inference Time (ms)": if_inf_time
    })
    
    results_df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/evaluation_results.csv", index=False)
    logger.info("Saved evaluation results to reports/evaluation_results.csv")
    
    # Save runs to SQLite DB
    try:
        import datetime
        from src.database.database import get_db_session, ModelRun
        with get_db_session() as session:
            session.add(ModelRun(
                model="rolling_zscore",
                trained_at=datetime.datetime.utcnow().isoformat(),
                precision=float(b_prec),
                recall=float(b_rec),
                f1=float(b_f1),
                pr_auc=float(b_pr_auc),
                roc_auc=float(b_roc_auc),
                threshold=float(k),
                notes="Tuned window=15 rolling z-score baseline"
            ))
            session.add(ModelRun(
                model="isolation_forest",
                trained_at=datetime.datetime.utcnow().isoformat(),
                precision=float(if_prec),
                recall=float(if_rec),
                f1=float(if_f1),
                pr_auc=float(if_pr_auc),
                roc_auc=float(if_roc_auc),
                threshold=float(thresh),
                notes="Tuned Isolation Forest on engineered causal features"
            ))
        logger.info("Saved model runs to database table 'model_runs'")
    except Exception as db_e:
        logger.warning(f"Could not persist model run details to DB: {db_e}")
    
    # Plotting using matplotlib
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        os.makedirs("reports/figures", exist_ok=True)
        
        # 1. raw_series_with_windows.png
        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["value"], label="Temperature", color="dodgerblue", alpha=0.8)
        
        from src.data.preprocessing import ANOMALY_WINDOWS
        first = True
        for start, end in ANOMALY_WINDOWS:
            plt.axvspan(pd.to_datetime(start).to_pydatetime(), pd.to_datetime(end).to_pydatetime(), color="red", alpha=0.3, label="Anomaly Window" if first else "")  # type: ignore[arg-type]
            first = False
            
        plt.title("NAB Machine Temperature Stream with Labeled Anomaly Windows")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/figures/raw_series_with_windows.png")
        plt.close()
        
        # 2. anomaly_score_timeline.png
        plt.figure(figsize=(12, 5))
        test_clean_times = test_clean["timestamp"].to_numpy()
        plt.plot(test_clean_times, if_scores, label="Isolation Forest Score", color="purple")
        plt.axhline(y=thresh, color="red", linestyle="--", label=f"Threshold ({thresh:.4f})")
        
        flagged_idx = np.where(if_preds == 1)[0]
        if len(flagged_idx) > 0:
            plt.scatter(test_clean_times[flagged_idx], if_scores[flagged_idx], color="red", label="Flagged Anomaly", zorder=5)  # type: ignore[index]
            
        plt.title("Isolation Forest Anomaly Score vs Decision Threshold (Test Set)")
        plt.xlabel("Timestamp")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/figures/anomaly_score_timeline.png")
        plt.close()
        
        # 3. confusion_matrix_if.png
        plt.figure(figsize=(5, 4))
        sns.heatmap(if_conf, annot=True, fmt="d", cmap="Purples", cbar=False,
                    xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        plt.title("Isolation Forest Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig("reports/figures/confusion_matrix_if.png")
        plt.close()
        
        # 4. pr_curve.png
        plt.figure(figsize=(6, 5))
        if if_pr_recall is not None and if_pr_precision is not None:
            plt.plot(if_pr_recall, if_pr_precision, label=f"Isolation Forest (AUC={if_pr_auc:.3f})", color="purple")
        if b_pr_recall is not None and b_pr_precision is not None:
            plt.plot(b_pr_recall, b_pr_precision, label=f"Rolling Z-score (AUC={b_pr_auc:.3f})", color="gray", linestyle="--")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/figures/pr_curve.png")
        plt.close()
        
        # 5. roc_curve.png
        from sklearn.metrics import roc_curve
        if_fpr, if_tpr, _ = roc_curve(y_test, if_scores)
        b_fpr, b_tpr, _ = roc_curve(y_test, zscores_test)
        plt.figure(figsize=(6, 5))
        plt.plot(if_fpr, if_tpr, label=f"Isolation Forest (AUC={if_roc_auc:.3f})", color="purple")
        plt.plot(b_fpr, b_tpr, label=f"Rolling Z-score (AUC={b_roc_auc:.3f})", color="gray", linestyle="--")
        plt.plot([0, 1], [0, 1], color="black", linestyle=":")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/figures/roc_curve.png")
        plt.close()
        
        # 6. model_comparison_bar.png
        plt.figure(figsize=(8, 4))
        comparison_melted = results_df.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1"])
        sns.barplot(data=comparison_melted, x="variable", y="value", hue="Model", palette="Set2")
        plt.ylim(0, 1.1)
        plt.title("Model Comparison: Precision, Recall, and F1-Score")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig("reports/figures/model_comparison_bar.png")
        plt.close()
        
        logger.info("Saved all evaluation figures to reports/figures/")
    except Exception as img_e:
        logger.error(f"Failed to generate evaluation plots: {img_e}")

if __name__ == "__main__":
    evaluate_models()
