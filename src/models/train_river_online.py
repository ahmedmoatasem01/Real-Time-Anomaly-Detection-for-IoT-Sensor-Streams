import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from river import anomaly

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("train_river")
settings = get_settings()

class RiverHSTWrapper:
    """
    Wraps a river HalfSpaceTrees model so it acts like an sklearn outlier detector
    for compatibility with evaluate_all.py and inference_service.py.
    """
    def __init__(self, model):
        self.model = model
        
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        scores = []
        # Iterate over rows
        for _, row in X.iterrows():
            x = row.to_dict()
            # river score_one returns probability of anomaly (higher = anomaly)
            # evaluate_all.py negates the output of score_samples: 
            # `scores = -model.score_samples(X_df)`
            # So we return -score to match sklearn convention.
            score = self.model.score_one(x)
            scores.append(-score)
            
            # NOTE: We do NOT call learn_one here during evaluation to strictly 
            # maintain the "fit on train-normal, score on test" paradigm and prevent leakage.
        return np.array(scores)


def train_river():
    logger.info("Starting River HalfSpaceTrees training (online model simulation)...")
    
    df = pd.read_csv(settings.PROCESSED_CSV)
    
    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
        
    # Filter training data (normal only)
    train_clean = df[(df["split"] == "train") & (df["label"] == 0)].dropna(subset=feature_cols)
    X_train_df = train_clean[feature_cols]
    
    val_clean = df[df["split"] == "val"].dropna(subset=feature_cols)
    X_val_df = val_clean[feature_cols]
    y_val = val_clean["label"].values
    
    # Initialize HalfSpaceTrees
    hst = anomaly.HalfSpaceTrees(
        n_trees=50,
        height=10,
        window_size=250,
        seed=42
    )
    
    logger.info(f"Training River HST sequentially on {len(X_train_df)} train samples...")
    for _, row in X_train_df.iterrows():
        x = row.to_dict()
        hst.learn_one(x)
        
    # Wrap model
    wrapper = RiverHSTWrapper(hst)
    
    # Save model artifact
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_DIR, "river_online.pkl")
    joblib.dump(wrapper, model_path)
    logger.info(f"Saved River wrapper to {model_path}")
    
    # Tuning threshold on validation set
    logger.info("Tuning threshold on validation set...")
    # val_scores will be negative of the river score
    val_scores_sklearn = wrapper.score_samples(X_val_df)
    
    # Revert to positive score for thresholding logic
    val_scores = -val_scores_sklearn
    
    best_thresh = None
    best_f1 = 0
    best_prec = 0
    best_rec = 0
    
    percentiles = np.linspace(80, 99.9, 100)
    for p in percentiles:
        thresh = np.percentile(val_scores, p)
        preds = (val_scores > thresh).astype(int)
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0.0
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_prec = prec
            best_rec = rec
            
    if best_thresh is None:
        best_thresh = float(np.percentile(val_scores, 95.0))
        
    threshold_data = {
        "threshold": float(best_thresh),
        "split": "validation",
        "selection_metrics": {
            "Precision": float(best_prec),
            "Recall": float(best_rec),
            "F1": float(best_f1)
        },
        "notes": "River HalfSpaceTrees online model."
    }
    
    thresh_path = os.path.join(settings.MODEL_DIR, "threshold_river.json")
    with open(thresh_path, "w") as f:
        json.dump(threshold_data, f, indent=4)
        
    logger.info(f"Selected threshold={best_thresh:.6f}")
    logger.info(f"Validation metrics: Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1={best_f1:.4f}")
    logger.info(f"Saved threshold configuration to {thresh_path}")

if __name__ == "__main__":
    train_river()
