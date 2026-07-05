import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import make_features

def test_make_features():
    # Create 70 rows of data (to clear warmup period of window size 60)
    data = {
        "value_scaled": np.linspace(1.0, 70.0, 70)
    }
    df = pd.DataFrame(data)
    
    features = make_features(df, windows=[5])
    
    # Check features generated
    expected_cols = [
        "value_scaled", "lag_1", "lag_2", "lag_3", "roc",
        "roll_mean_5", "roll_std_5", "roll_min_5", "roll_max_5", "ewma_5", "zscore_5"
    ]
    for col in expected_cols:
        assert col in features.columns

    # Verify lag_1 shifts correctly
    assert features["lag_1"].iloc[1] == features["value_scaled"].iloc[0]
    assert features["lag_1"].iloc[10] == features["value_scaled"].iloc[9]
    
    # Verify rate of change (roc) is constant in linear space
    assert pytest.approx(features["roc"].iloc[2]) == 1.0
    
    # Verify roll_mean_5 at row 10 is average of values from row 6 to 10 (indices 6 to 10)
    expected_mean = np.mean(features["value_scaled"].iloc[6:11])
    assert pytest.approx(features["roll_mean_5"].iloc[10]) == expected_mean

def test_no_future_leakage():
    # Make sure features at index t do NOT depend on values at index > t
    data = {"value_scaled": np.random.randn(80)}
    df = pd.DataFrame(data)
    
    features_orig = make_features(df)
    
    # Modify values at the end of the series (indices 70-79)
    df_modified = df.copy()
    df_modified.loc[70:, "value_scaled"] = 999.0
    
    features_mod = make_features(df_modified)
    
    # Check that for indices < 70, the features are EXACTLY identical
    # This guarantees no lookahead or center window leakage
    pd.testing.assert_frame_equal(
        features_orig.iloc[:70],
        features_mod.iloc[:70]
    )

def test_feature_order_contract():
    import json
    import os
    from src.utils.config import get_settings
    
    settings = get_settings()
    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    
    assert os.path.exists(feature_cols_path), "Feature contract JSON does not exist!"
    
    with open(feature_cols_path, "r") as f:
        contract_cols = json.load(f)
        
    df = pd.DataFrame({"value_scaled": [1,2,3]})
    features = make_features(df)
    
    generated_cols = [c for c in features.columns if c != "value_scaled"]
    
    # Assert that all contract columns are in the generated columns
    for col in contract_cols:
        assert col in generated_cols, f"Column {col} missing from generated features"
        
    # We do not assert exact list equality because make_features can generate 
    # more columns (if more windows are passed), but for the default windows 
    # we assert the contract is a subset. Actually, let's assert exact order 
    # for the default ones.
    
    expected_order = [
        "lag_1", "lag_2", "lag_3", "roc"
    ]
    for w in [5, 15, 60]:
        expected_order.extend([
            f"roll_mean_{w}", f"roll_std_{w}", f"roll_min_{w}", f"roll_max_{w}", f"ewma_{w}", f"zscore_{w}"
        ])
        
    assert contract_cols == expected_order, "Contract does not match expected order!"
