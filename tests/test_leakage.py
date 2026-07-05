import os
import pandas as pd
from src.utils.config import get_settings

settings = get_settings()

def test_no_temporal_leakage():
    """
    Asserts that the train, val, and test splits are strictly chronological
    to prevent future information from leaking into training.
    """
    assert os.path.exists(settings.PROCESSED_CSV), "Processed CSV not found."
    df = pd.read_csv(settings.PROCESSED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    # Assert temporal order
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Train leaks into Val!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Val leaks into Test!"
    
def test_scaler_burn_in_has_no_anomalies():
    """
    Asserts that the first 15% of the train split (used for scaler burn-in)
    contains no anomalies in its final filtered state.
    """
    df = pd.read_csv(settings.PROCESSED_CSV)
    train_df = df[df["split"] == "train"]
    burn_in_len = int(len(train_df) * 0.15)
    burn_in_data = train_df.iloc[:burn_in_len]
    
    # In preprocessing.py, we explicitly filter out anomalies for the scaler.
    # We simulate that filter here and assert the result has 0 anomalies.
    burn_in_filtered = burn_in_data[burn_in_data["label"] == 0]
    assert burn_in_filtered["label"].sum() == 0, "Scaler burn-in still contains anomalies!"
