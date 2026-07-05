import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("train_baseline")
settings = get_settings()

def tune_baseline():
    logger.info("Starting baseline (rolling Z-score) tuning...")
    if not os.path.exists(settings.PROCESSED_CSV):
        raise FileNotFoundError(f"Processed CSV not found at {settings.PROCESSED_CSV}")
        
    df = pd.read_csv(settings.PROCESSED_CSV)
    
    # Filter validation set
    val_df = df[df["split"] == "val"].copy()
    
    # Drop rows where zscore_15 is NaN (though there shouldn't be any in val)
    val_df = val_df.dropna(subset=["zscore_15"])  # type: ignore[call-overload]
    
    if len(val_df) == 0:
        raise ValueError("Validation set is empty or contains only NaNs.")
        
    y_val = val_df["label"].values
    zscores = np.abs(val_df["zscore_15"].values)
    
    best_f1 = -1
    best_k = 3.0
    best_metrics = (0.0, 0.0, 0.0) # precision, recall, f1
    
    # Search for best k
    k_grid = np.linspace(1.0, 6.0, 51)
    for k in k_grid:
        preds = (zscores > k).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0.0  # type: ignore[call-overload]
        )
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_metrics = (precision, recall, f1)
            
    logger.info(f"Optimal baseline k found: {best_k:.2f}")
    logger.info(f"Validation metrics at best k: Precision={best_metrics[0]:.4f}, Recall={best_metrics[1]:.4f}, F1={best_metrics[2]:.4f}")
    
    # Save baseline threshold config
    threshold_path = os.path.join(settings.MODEL_DIR, "threshold_zscore.json")
    threshold_config = {
        "k": float(best_k),
        "window": 15,
        "model": "rolling_zscore",
        "precision": float(best_metrics[0]),
        "recall": float(best_metrics[1]),
        "f1": float(best_metrics[2])
    }
    with open(threshold_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Saved baseline threshold configuration to {threshold_path}")

if __name__ == "__main__":
    tune_baseline()
