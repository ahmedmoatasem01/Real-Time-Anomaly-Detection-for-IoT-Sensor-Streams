"""
Train One-Class SVM for anomaly detection on machine temperature data.
Fits on known-normal train rows; tunes nu on validation for best F1.
Artifacts saved: models/one_class_svm.pkl + models/threshold_ocsvm.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("train_one_class_svm")
settings = get_settings()


def train_ocsvm():
    logger.info("Starting One-Class SVM training...")

    if not os.path.exists(settings.PROCESSED_CSV):
        raise FileNotFoundError(f"Processed CSV not found at {settings.PROCESSED_CSV}")

    df = pd.read_csv(settings.PROCESSED_CSV)

    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    logger.info(f"Using {len(feature_cols)} feature columns from feature_columns.json")

    # Split sets
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()

    train_normal = train_df[train_df["label"] == 0]
    train_clean  = train_normal.dropna(subset=feature_cols)
    val_clean    = val_df.dropna(subset=feature_cols)

    X_train = train_clean[feature_cols]
    X_val   = val_clean[feature_cols]
    y_val   = val_clean["label"].values

    # Grid-search nu on validation
    best_f1      = -1.0
    best_nu      = 0.05
    best_model   = None
    best_metrics = (0.0, 0.0, 0.0)
    best_thresh  = 0.0

    nu_grid = [0.01, 0.03, 0.05, 0.1]
    for nu in nu_grid:
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        model.fit(X_train)

        # score = -decision_function → higher = more anomalous
        val_scores = -model.decision_function(X_val)
        # Tune threshold on validation PR
        thresh_grid = np.linspace(val_scores.min(), val_scores.max(), 100)
        for thresh in thresh_grid:
            preds = (val_scores > thresh).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1      = f1
                best_nu      = nu
                best_model   = model
                best_metrics = (prec, rec, f1)
                best_thresh  = thresh

        logger.info(f"nu={nu:.2f} -> best val F1 so far: {best_f1:.4f}")

    logger.info(f"Selected nu={best_nu:.2f} | threshold={best_thresh:.6f}")
    logger.info(
        f"Validation selection metrics: "
        f"Precision={best_metrics[0]:.4f}, Recall={best_metrics[1]:.4f}, F1={best_metrics[2]:.4f}"
    )

    # Save artifacts
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_DIR, "one_class_svm.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Saved One-Class SVM to {model_path}")

    threshold_config = {
        "threshold": float(best_thresh),
        "model": "one_class_svm",
        "nu": float(best_nu),
        "selection_metrics": {
            "split": "validation",
            "note": "Used for threshold selection only. See evaluation_results.csv for test metrics.",
            "precision": float(best_metrics[0]),
            "recall":    float(best_metrics[1]),
            "f1":        float(best_metrics[2]),
        },
    }
    thresh_path = os.path.join(settings.MODEL_DIR, "threshold_ocsvm.json")
    with open(thresh_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Saved OCSVM threshold (validation selection metrics) to {thresh_path}")


if __name__ == "__main__":
    train_ocsvm()
