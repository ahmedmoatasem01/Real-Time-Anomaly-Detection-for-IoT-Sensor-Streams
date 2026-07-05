import os
import json
import pandas as pd
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")
settings = get_settings()

def make_features(df: pd.DataFrame, windows=[5, 15, 60], value_col="value_scaled") -> pd.DataFrame:
    """
    Computes causal time-series features based on value_col.
    No future leakage: only uses current and past observations.
    """
    df = df.copy()
    
    # Non-windowed features
    df["lag_1"] = df[value_col].shift(1)
    df["lag_2"] = df[value_col].shift(2)
    df["lag_3"] = df[value_col].shift(3)
    df["roc"] = df[value_col].diff()
    
    # Windowed features
    for w in windows:
        df[f"roll_mean_{w}"] = df[value_col].rolling(window=w).mean()
        df[f"roll_std_{w}"] = df[value_col].rolling(window=w).std()
        df[f"roll_min_{w}"] = df[value_col].rolling(window=w).min()
        df[f"roll_max_{w}"] = df[value_col].rolling(window=w).max()
        df[f"ewma_{w}"] = df[value_col].ewm(span=w, adjust=False).mean()
        
        # Avoid division by zero with small epsilon
        df[f"zscore_{w}"] = (df[value_col] - df[f"roll_mean_{w}"]) / (df[f"roll_std_{w}"] + 1e-9)
        
    return df

def feature_engineering_pipeline():
    logger.info("Starting feature engineering pipeline...")
    if not os.path.exists(settings.PROCESSED_CSV):
        raise FileNotFoundError(f"Processed CSV not found at {settings.PROCESSED_CSV}")
        
    df = pd.read_csv(settings.PROCESSED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Generate features
    df_features = make_features(df)
    
    # Save the order of feature columns for model consistency
    feature_cols = [
        "lag_1", "lag_2", "lag_3", "roc"
    ]
    for w in [5, 15, 60]:
        feature_cols.extend([
            f"roll_mean_{w}", f"roll_std_{w}", f"roll_min_{w}", f"roll_max_{w}", f"ewma_{w}", f"zscore_{w}"
        ])
        
    # Let's save feature columns definition
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f)
    logger.info(f"Saved feature column names to {feature_cols_path}")
    
    # Write full dataframe with engineered features
    df_features.to_csv(settings.PROCESSED_CSV, index=False)
    logger.info(f"Updated processed CSV with features at {settings.PROCESSED_CSV}")

if __name__ == "__main__":
    feature_engineering_pipeline()
