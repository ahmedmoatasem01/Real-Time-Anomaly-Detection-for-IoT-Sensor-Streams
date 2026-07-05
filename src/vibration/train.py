import os
import json
import joblib
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.vibration.data_loader import get_snapshot_files, load_vibration_snapshot
from src.vibration.features import extract_all_features

logger = get_logger("vibration_train")

VIBRATION_DIR = "data/raw/bearing/2nd_test" # Path for NASA Bearing Test 2
MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

def load_and_extract_features(data_dir: str, num_files_to_process: int = None) -> pd.DataFrame:
    """
    Loads raw snapshots and extracts features into a DataFrame.
    """
    files = get_snapshot_files(data_dir)
    if not files:
        logger.error(f"No files found in {data_dir}. Ensure NASA Bearing Test 2 is downloaded.")
        return pd.DataFrame()
        
    if num_files_to_process:
        # e.g. subset for faster testing
        files = files[::max(1, len(files)//num_files_to_process)]
        
    logger.info(f"Processing {len(files)} files for feature extraction...")
    
    rows = []
    for idx, f in enumerate(files):
        # Bearing 1 is known to fail in Test 2
        signal = load_vibration_snapshot(f, bearing_idx=0)
        if len(signal) > 0:
            features = extract_all_features(signal)
            
            # The filename is the timestamp, let's parse it if needed, or just keep it as ID
            basename = os.path.basename(f)
            features["timestamp"] = basename
            features["file_index"] = idx
            
            rows.append(features)
            
    df = pd.DataFrame(rows)
    # Reorder so timestamp is first
    cols = ["timestamp", "file_index"] + [c for c in df.columns if c not in ["timestamp", "file_index"]]
    df = df[cols]
    
    logger.info(f"Extracted features for {len(df)} snapshots.")
    return df

def update_registry(model_name: str, metrics: dict):
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = []
        
    entry = {
        "name": model_name,
        "modality": "vibration",
        "dataset": "NASA Bearing Test 2",
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "metrics": metrics,
        "is_production": True,
        "artifact_path": f"{MODELS_DIR}/{model_name}.pkl"
    }
    
    # Update if exists, else append
    updated = False
    for i, mod in enumerate(registry):
        if mod["name"] == model_name:
            registry[i] = entry
            updated = True
            break
            
    if not updated:
        registry.append(entry)
        
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)
    logger.info(f"Registered model {model_name} in {REGISTRY_PATH}")


def train_models():
    """
    Extracts features, trains Isolation Forest on normal data, and saves artifacts.
    """
    df = load_and_extract_features(VIBRATION_DIR)
    if df.empty:
        return
        
    # The first 400 files in Test 2 are considered "normal" (healthy bearing)
    # The failure occurs near the end of the ~984 files.
    train_df = df.iloc[:400].copy()
    test_df = df.iloc[400:].copy()
    
    feature_cols = [c for c in df.columns if c not in ["timestamp", "file_index"]]
    
    logger.info(f"Training on {len(train_df)} normal samples with features: {feature_cols}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    
    # Train Isolation Forest
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(X_train)
    
    # Score on test set to see degradation
    X_test = scaler.transform(test_df[feature_cols])
    scores = clf.decision_function(X_test)
    
    logger.info(f"Test scores range: {scores.min():.4f} to {scores.max():.4f}")
    # Lower score = more anomalous. Near the end, scores should be highly negative.
    
    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "vibration_scaler.pkl"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "vibration_iforest.pkl"))
    
    # Save feature columns
    with open(os.path.join(MODELS_DIR, "vibration_feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=4)
        
    # Register
    metrics = {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "min_score": float(scores.min()),
        "max_score": float(scores.max())
    }
    update_registry("vibration_iforest", metrics)
    logger.info("Vibration training complete.")

if __name__ == "__main__":
    train_models()
