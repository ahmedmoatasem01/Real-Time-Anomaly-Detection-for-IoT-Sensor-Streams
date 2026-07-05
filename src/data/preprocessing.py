import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.data_loader import load_nab
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("preprocessing")
settings = get_settings()

# NAB anomaly windows for machine_temperature_system_failure
ANOMALY_WINDOWS = [
    ("2013-12-10 06:25:00", "2013-12-12 05:35:00"),
    ("2013-12-15 17:50:00", "2013-12-17 17:00:00"),
    ("2014-01-27 14:20:00", "2014-01-29 13:30:00"),
    ("2014-02-07 14:55:00", "2014-02-09 14:05:00")
]

def add_anomaly_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds labels based on historical anomaly windows.
    """
    df = df.copy()
    df["label"] = 0
    for start, end in ANOMALY_WINDOWS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df.loc[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt), "label"] = 1
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe, handles missing values and duplicates.
    """
    df = df.copy()
    
    # 1. Parse timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 2. Rename value column if needed
    if "value" not in df.columns:
        float_cols = df.select_dtypes(include=[np.number]).columns
        if len(float_cols) > 0:
            col_name = str(float_cols[0])
            df = df.rename(columns={col_name: "value"})  # type: ignore[call-overload]
            logger.info(f"Renamed column '{col_name}' to 'value'")
            
    # 3. Drop duplicate timestamps, keeping last
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    
    # 4. Handle missing values by reindexing to a complete 5-minute grid
    df = df.set_index("timestamp")
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
    
    # Flag imputed values
    df_reindexed = df.reindex(full_index)
    df_reindexed["imputed"] = df_reindexed["value"].isna().astype(int)
    
    # Pre-calculate NaN block sizes for logic:
    # ffill for gaps <= 3; linear interpolate for gaps > 3
    is_na = df_reindexed["value"].isna()
    nan_block = (is_na != is_na.shift()).cumsum()
    block_sizes = df_reindexed.groupby(nan_block)["value"].transform("size")
    
    ffilled = df_reindexed["value"].ffill()
    interpolated = df_reindexed["value"].interpolate(method="linear")
    
    df_reindexed["value"] = np.where(
        ~is_na,
        df_reindexed["value"],
        np.where(block_sizes <= 3, ffilled, interpolated)
    )
    
    # Reset index and restore timestamp column
    df_reindexed = df_reindexed.reset_index().rename(columns={"index": "timestamp"})
    return df_reindexed

def split_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits data chronologically: 70% train, 15% val, 15% test.
    Adds a 'split' column to the dataframe.
    """
    df = df.copy()
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    df["split"] = "train"
    df.loc[train_end:val_end, "split"] = "val"
    df.loc[val_end:, "split"] = "test"
    
    return df

def preprocess_pipeline():
    logger.info("Starting preprocessing pipeline...")
    
    # 1. Load raw data
    raw_df = load_nab()
    logger.info(f"Loaded raw data with shape: {raw_df.shape}")
    
    # 2. Clean and handle missing values
    cleaned_df = clean_data(raw_df)
    logger.info(f"Cleaned data. New shape: {cleaned_df.shape}")
    
    # 3. Add sensor ID
    cleaned_df["sensor_id"] = settings.SENSOR_ID
    
    # 4. Add labels
    labeled_df = add_anomaly_labels(cleaned_df)
    logger.info(f"Added anomaly labels. Total anomalies: {labeled_df['label'].sum()}")
    
    # 5. Split chronologically
    split_df = split_data(labeled_df)
    logger.info(f"Splits summary: {split_df['split'].value_counts().to_dict()}")
    
    # 6. Fit StandardScaler on normal training data (first 15% of train split)
    train_df = split_df[split_df["split"] == "train"]
    burn_in_len = int(len(train_df) * 0.15)
    burn_in_data = train_df.iloc[:burn_in_len]
    
    if burn_in_data["label"].sum() > 0:
        logger.warning("Burn-in data contains anomalies! Filtering to label=0.")
        burn_in_data = burn_in_data[burn_in_data["label"] == 0]
        
    scaler = StandardScaler()
    scaler.fit(burn_in_data[["value"]])
    logger.info(f"Fit StandardScaler. Mean: {float(scaler.mean_[0])}, Scale: {float(scaler.scale_[0])}")  # type: ignore[index]
    
    # Save scaler
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(settings.MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # 7. Scale values
    split_df["value_scaled"] = scaler.transform(split_df[["value"]])
    
    # 8. Save processed data
    os.makedirs(os.path.dirname(settings.PROCESSED_CSV), exist_ok=True)
    split_df.to_csv(settings.PROCESSED_CSV, index=False)
    logger.info(f"Saved processed CSV to {settings.PROCESSED_CSV}")
    
    # 9. Save test stream sample
    test_df = split_df[split_df["split"] == "test"]
    os.makedirs("data/sample_stream", exist_ok=True)
    test_stream_path = "data/sample_stream/test_stream.csv"
    test_df.to_csv(test_stream_path, index=False)
    logger.info(f"Saved test stream sample to {test_stream_path}")
    
if __name__ == "__main__":
    preprocess_pipeline()
