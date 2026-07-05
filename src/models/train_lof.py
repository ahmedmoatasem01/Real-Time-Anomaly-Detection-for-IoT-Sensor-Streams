"""
Train Local Outlier Factor (novelty=True) for machine temperature anomaly detection.
Fits on known-normal train rows; tunes n_neighbors on validation for best F1.
Artifacts saved: models/lof.pkl + models/threshold_lof.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("train_lof")
settings = get_settings()


def train_lof():
    logger.info("Starting LOF (novelty mode) training...")

    if not os.path.exists(settings.PROCESSED_CSV):
        raise FileNotFoundError(f"Processed CSV not found at {settings.PROCESSED_CSV}")

    df = pd.read_csv(settings.PROCESSED_CSV)

    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    logger.info(f"Using {len(feature_cols)} feature columns from feature_columns.json")

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()

    train_normal = train_df[train_df["label"] == 0]
    train_clean  = train_normal.dropna(subset=feature_cols)  # type: ignore[call-overload]
    val_clean    = val_df.dropna(subset=feature_cols)  # type: ignore[call-overload]

    X_train = train_clean[feature_cols]
    X_val = val_clean[feature_cols]
    y_val   = val_clean["label"].values

    best_f1      = -1.0
    best_k       = 20
    best_model   = None
    best_metrics = (0.0, 0.0, 0.0)
    best_thresh  = 0.0

    k_grid = [20, 35, 50]
    for k in k_grid:
        model = LocalOutlierFactor(n_neighbors=k, novelty=True)
        model.fit(X_train)

        # score_samples → higher (less negative) = more normal → negate for "anomaly score"
        val_scores = -model.score_samples(X_val)
        thresh_grid = np.linspace(val_scores.min(), val_scores.max(), 100)
        for thresh in thresh_grid:
            preds = (val_scores > thresh).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, preds, average="binary", zero_division=0.0  # type: ignore[call-overload]
            )
            if f1 > best_f1:
                best_f1      = f1
                best_k       = k
                best_model   = model
                best_metrics = (prec, rec, f1)
                best_thresh  = thresh

        logger.info(f"n_neighbors={k} -> best val F1 so far: {best_f1:.4f}")

    logger.info(f"Selected n_neighbors={best_k} | threshold={best_thresh:.6f}")
    logger.info(
        f"Validation selection metrics: "
        f"Precision={best_metrics[0]:.4f}, Recall={best_metrics[1]:.4f}, F1={best_metrics[2]:.4f}"
    )

    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_DIR, "lof.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Saved LOF model to {model_path}")

    threshold_config = {
        "threshold": float(best_thresh),
        "model": "lof",
        "n_neighbors": int(best_k),
        "selection_metrics": {
            "split": "validation",
            "note": "Used for threshold selection only. See evaluation_results.csv for test metrics.",
            "precision": float(best_metrics[0]),
            "recall":    float(best_metrics[1]),
            "f1":        float(best_metrics[2]),
        },
    }
    thresh_path = os.path.join(settings.MODEL_DIR, "threshold_lof.json")
    with open(thresh_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Saved LOF threshold (validation selection metrics) to {thresh_path}")


if __name__ == "__main__":
    train_lof()
