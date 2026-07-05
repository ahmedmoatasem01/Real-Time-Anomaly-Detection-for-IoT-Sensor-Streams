import os
import subprocess
import pandas as pd
from sqlalchemy.orm import Session
from src.database.database import SessionLocal, Reading
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.registry.model_registry import load_registry
import json

logger = get_logger("retrain_pipeline")
settings = get_settings()

def export_recent_data_to_csv():
    db = SessionLocal()
    try:
        # Export all readings to a temporary CSV or just overwrite data/processed/sensor_features.csv?
        # Let's write directly to settings.PROCESSED_CSV to make it simple for the training scripts to pick up
        readings = db.query(Reading).order_by(Reading.id.desc()).limit(10000).all()
        readings = readings[::-1] # chronological
        if not readings:
            logger.warning("No data found for retraining.")
            return False
            
        data = []
        for r in readings:
            # We need to format it like sensor_features.csv
            # But wait, original sensor_features.csv has lots of features (zscore_15, rolling_mean, etc.)
            # We don't have all of them in Reading, but we can just use value as a proxy or reconstruct them.
            data.append({
                "ts": r.ts,
                "sensor_id": r.sensor_id,
                "value": r.value,
                "label": int(r.is_anomaly),
                "split": "test" if len(data) > 8000 else "train" # 80% train, 20% test
            })
            
        df = pd.DataFrame(data)
        # Compute zscore_15 for the baseline model
        df["rolling_mean"] = df["value"].rolling(15, min_periods=1).mean()
        df["rolling_std"] = df["value"].rolling(15, min_periods=1).std().fillna(1e-9)
        df["zscore_15"] = (df["value"] - df["rolling_mean"]) / df["rolling_std"]
        
        # Add a couple mock features if needed by isolation forest
        df["value_diff_1"] = df["value"].diff().fillna(0)
        df["value_diff_2"] = df["value_diff_1"].diff().fillna(0)
        df["fft_component_1"] = df["value"] * 0.1 # dummy
        
        os.makedirs(os.path.dirname(settings.PROCESSED_CSV), exist_ok=True)
        df.to_csv(settings.PROCESSED_CSV, index=False)
        
        # Make sure feature_columns.json matches
        feature_cols = ["value", "zscore_15", "value_diff_1", "value_diff_2"]
        with open(os.path.join(settings.MODEL_DIR, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
            
        return True
    finally:
        db.close()

def run_retraining():
    logger.info("Starting retraining pipeline...")
    
    # 1. Export data
    if not export_recent_data_to_csv():
        return False
        
    # 2. Train models
    logger.info("Training Isolation Forest...")
    subprocess.run([".venv\\Scripts\\python.exe", "-m", "src.models.train_isolation_forest"], check=False)
    
    # 3. Evaluate models
    logger.info("Evaluating models...")
    subprocess.run([".venv\\Scripts\\python.exe", "-m", "src.models.evaluate_all"], check=False)
    
    # 4. Find the best model in evaluation_results.csv
    if not os.path.exists("reports/evaluation_results.csv"):
        logger.error("Evaluation results not found.")
        return False
        
    results_df = pd.read_csv("reports/evaluation_results.csv")
    if len(results_df) == 0:
        return False
        
    best_model_name = results_df.iloc[0]["Model"]
    
    # Get current production model
    registry = load_registry()
    current_prod = next((k for k, v in registry.items() if v.get("is_production")), None)
    
    if current_prod != best_model_name:
        logger.info(f"Promoting {best_model_name} to production (replacing {current_prod}).")
        # In actual system, we'd hit /models/select/{name} but we can just use registry directly
        from src.registry.model_registry import set_production
        from src.api.inference_service import get_inference_service
        set_production(best_model_name)
        
        # Refresh inference service
        svc = get_inference_service()
        svc.load_artifacts()
        
    return True
