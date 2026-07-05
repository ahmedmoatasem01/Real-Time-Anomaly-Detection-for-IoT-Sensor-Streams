import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("train_isolation_forest")
settings = get_settings()

def train_iforest():
    logger.info("Starting Isolation Forest training...")
    if not os.path.exists(settings.PROCESSED_CSV):
        raise FileNotFoundError(f"Processed CSV not found at {settings.PROCESSED_CSV}")
        
    df = pd.read_csv(settings.PROCESSED_CSV)
    
    # Load feature columns
    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
        
    logger.info(f"Using {len(feature_cols)} feature columns: {feature_cols}")
    
    # Filter splits
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    
    # Filter train to normal only (unsupervised setup)
    train_normal = train_df[train_df["label"] == 0]
    
    # Drop rows with NaNs in feature columns (warmup period)
    train_clean = train_normal.dropna(subset=feature_cols)  # type: ignore[call-overload]
    val_clean = val_df.dropna(subset=feature_cols)  # type: ignore[call-overload]
    
    X_train = train_clean[feature_cols]
    
    # Initialize Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    
    logger.info(f"Fitting Isolation Forest on {len(X_train)} training rows...")
    model.fit(X_train)
    
    # Save model
    model_path = os.path.join(settings.MODEL_DIR, "isolation_forest.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved Isolation Forest model to {model_path}")
    
    # Determine the optimal threshold on validation set
    X_val = val_clean[feature_cols]
    y_val = val_clean["label"].values
    
    # Anomaly score is -model.score_samples()
    # We want score = -model.score_samples() so that higher is more anomalous
    val_scores = -model.score_samples(X_val)
    
    best_f1 = -1
    best_thresh = 0.5
    best_metrics = (0.0, 0.0, 0.0)
    
    # Grid search threshold on validation set
    thresh_grid = np.linspace(val_scores.min(), val_scores.max(), 100)
    for thresh in thresh_grid:
        preds = (val_scores > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0.0  # type: ignore[call-overload]
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = (precision, recall, f1)
            
    logger.info(f"Optimal Isolation Forest threshold found: {best_thresh:.6f}")
    logger.info(f"Validation metrics: Precision={best_metrics[0]:.4f}, Recall={best_metrics[1]:.4f}, F1={best_metrics[2]:.4f}")
    
    # Save feature stats (mean and std) on training set for explanation
    feature_stats = {}
    for col in feature_cols:
        feature_stats[col] = {
            "mean": float(train_clean[col].mean()),
            "std": float(train_clean[col].std())
        }
    stats_path = os.path.join(settings.MODEL_DIR, "feature_stats.json")
    with open(stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    logger.info(f"Saved feature stats to {stats_path}")

    # Save threshold with clearly-labeled selection_metrics (chosen on validation).
    # NOTE: These are VALIDATION metrics used ONLY for threshold selection.
    # Authoritative TEST metrics are written exclusively by evaluate_all.py.
    threshold_path = os.path.join(settings.MODEL_DIR, "threshold.json")
    threshold_config = {
        "threshold": float(best_thresh),
        "model": "isolation_forest",
        "selection_metrics": {
            "split": "validation",
            "note": "Used for threshold selection only. See evaluation_results.csv for test metrics.",
            "precision": float(best_metrics[0]),
            "recall": float(best_metrics[1]),
            "f1": float(best_metrics[2]),
        },
    }
    with open(threshold_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Saved threshold configuration (validation selection metrics) to {threshold_path}")

if __name__ == "__main__":
    train_iforest()
