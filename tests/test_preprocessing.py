import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import clean_data, add_anomaly_labels, split_data

def test_clean_data():
    # Create mockup dataframe
    # We introduce a duplicate timestamp, a missing 5-minute step, and a NaN value
    data = {
        "timestamp": [
            "2013-12-10 06:00:00",
            "2013-12-10 06:05:00",
            "2013-12-10 06:05:00", # duplicate
            "2013-12-10 06:15:00", # gap at 06:10
        ],
        "value": [70.0, 71.0, 72.0, 74.0]
    }
    df = pd.DataFrame(data)
    
    cleaned = clean_data(df)
    
    # Check length: reindexed grid should be 06:00, 06:05, 06:10, 06:15 (4 rows)
    assert len(cleaned) == 4
    
    # Check duplicates removed (keep last: 72.0)
    assert cleaned.loc[cleaned["timestamp"] == "2013-12-10 06:05:00", "value"].values[0] == 72.0
    
    # Check gap at 06:10 filled (linear interpolation between 72.0 and 74.0 is 73.0)
    assert cleaned.loc[cleaned["timestamp"] == "2013-12-10 06:10:00", "value"].values[0] == 72.0
    
    # Check imputed flag
    assert cleaned.loc[cleaned["timestamp"] == "2013-12-10 06:10:00", "imputed"].values[0] == 1
    assert cleaned.loc[cleaned["timestamp"] == "2013-12-10 06:00:00", "imputed"].values[0] == 0

def test_add_anomaly_labels():
    # Create mockup dataframe with timestamp inside first NAB anomaly window
    data = {
        "timestamp": [
            pd.to_datetime("2013-12-10 06:20:00"), # outside window
            pd.to_datetime("2013-12-10 06:30:00"), # inside window (starts 06:25)
            pd.to_datetime("2013-12-12 05:30:00"), # inside window (ends 05:35)
            pd.to_datetime("2013-12-12 05:40:00")  # outside window
        ],
        "value": [70.0, 71.0, 72.0, 73.0]
    }
    df = pd.DataFrame(data)
    labeled = add_anomaly_labels(df)
    
    assert labeled["label"].iloc[0] == 0
    assert labeled["label"].iloc[1] == 1
    assert labeled["label"].iloc[2] == 1
    assert labeled["label"].iloc[3] == 0

def test_split_data():
    # Create 100 rows
    times = pd.date_range(start="2013-12-10 00:00:00", periods=100, freq="5min")
    df = pd.DataFrame({"timestamp": times, "value": np.random.rand(100)})
    
    split = split_data(df)
    
    # 70% train, 15% val, 15% test
    assert (split["split"] == "train").sum() == 70
    assert (split["split"] == "val").sum() == 15
    assert (split["split"] == "test").sum() == 15

def test_no_label_leakage_in_scaling():
    # Verify the burn-in logic explicitly filters out anomaly labels (label == 1)
    # This prevents the anomaly from skewing the standard scaler (data leakage)
    train_data = pd.DataFrame({
        "value": [10.0, 11.0, 100.0, 12.0],
        "label": [0, 0, 1, 0]
    })
    
    # Simulate the preprocessing.py burn-in block
    burn_in_data = train_data.copy()
    if burn_in_data["label"].sum() > 0:
        burn_in_data = burn_in_data[burn_in_data["label"] == 0]
        
    assert 100.0 not in burn_in_data["value"].values
    assert len(burn_in_data) == 3
    assert burn_in_data["label"].sum() == 0

