"""
Train Elliptic Envelope for machine temperature anomaly detection.
Fits on known-normal train rows; handles singular covariance gracefully.
Artifacts saved: models/elliptic_envelope.pkl + models/threshold_elliptic.json
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import precision_recall_fscore_support
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("train_elliptic_envelope")
settings = get_settings()


def train_elliptic():
    logger.info("Starting Elliptic Envelope training...")

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

    # Try standard fit; if covariance is singular fall back to support_fraction
    model = None
    notes = ""
    for support_fraction in [None, 0.95, 0.90, 0.85]:
        try:
            kwargs = dict(contamination=0.02, random_state=42)
            if support_fraction is not None:
                kwargs["support_fraction"] = support_fraction
                logger.warning(
                    f"Singular covariance detected — retrying with support_fraction={support_fraction}"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                candidate = EllipticEnvelope(**kwargs)  # type: ignore[arg-type]
                candidate.fit(X_train)
            model = candidate
            notes = f"support_fraction={support_fraction}" if support_fraction else "default"
            logger.info(f"Elliptic Envelope fitted successfully ({notes})")
            break
        except Exception as e:
            logger.warning(f"Fit failed with support_fraction={support_fraction}: {e}")
            continue

    if model is None:
        logger.error(
            "Elliptic Envelope could not be fitted on this dataset "
            "(likely near-singular covariance for 22-dimensional data). "
            "This is a real finding — the model is not suitable here. Skipping."
        )
        return

    # Tune threshold on validation
    val_scores = -model.score_samples(X_val)
    best_f1      = -1.0
    best_thresh  = 0.0
    best_metrics = (0.0, 0.0, 0.0)

    thresh_grid = np.linspace(val_scores.min(), val_scores.max(), 100)
    for thresh in thresh_grid:
        preds = (val_scores > thresh).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0.0  # type: ignore[call-overload]
        )
        if f1 > best_f1:
            best_f1      = f1
            best_thresh  = thresh
            best_metrics = (prec, rec, f1)

    logger.info(f"Optimal threshold: {best_thresh:.6f}")
    logger.info(
        f"Validation selection metrics: "
        f"Precision={best_metrics[0]:.4f}, Recall={best_metrics[1]:.4f}, F1={best_metrics[2]:.4f}"
    )

    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_DIR, "elliptic_envelope.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved Elliptic Envelope to {model_path}")

    threshold_config = {
        "threshold": float(best_thresh),
        "model": "elliptic_envelope",
        "notes": notes,
        "selection_metrics": {
            "split": "validation",
            "note": "Used for threshold selection only. See evaluation_results.csv for test metrics.",
            "precision": float(best_metrics[0]),
            "recall":    float(best_metrics[1]),
            "f1":        float(best_metrics[2]),
        },
    }
    thresh_path = os.path.join(settings.MODEL_DIR, "threshold_elliptic.json")
    with open(thresh_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Saved Elliptic Envelope threshold (validation selection metrics) to {thresh_path}")


if __name__ == "__main__":
    train_elliptic()
