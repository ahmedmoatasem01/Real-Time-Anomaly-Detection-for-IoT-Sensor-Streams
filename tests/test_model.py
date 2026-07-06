import pytest
import os
import json
import joblib
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.api.inference_service import InferenceService

def test_inference_service_fallback():
    # When model/scaler are not loaded/trained, verify it returns standard dict
    service = InferenceService()
    # Mock them to None to guarantee fallback triggers
    service.model = None
    service.scaler = None
    
    res = service.predict(
        timestamp="2014-01-27T14:25:00",
        sensor_id="machine_temperature",
        value=72.13
    )
    
    assert res["is_anomaly"] is False
    assert res["anomaly_score"] == 0.0
    assert res["severity"] == "none"
    assert "not loaded" in res["reason"]

def test_explain_anomaly():
    service = InferenceService()
    # Inject mocked feature stats for testing
    service.feature_stats = {
        "roll_std_15": {"mean": 0.2, "std": 0.05},
        "zscore_15": {"mean": 0.0, "std": 1.0}
    }
    
    # Feature vector with normal zscore, but anomalous roll_std_15
    feature_vector = {
        "roll_std_15": 1.2, # deviation of (1.2 - 0.2) / 0.05 = 20.0 stds!
        "zscore_15": 0.5    # deviation of (0.5 - 0.0) / 1.0 = 0.5 stds
    }
    
    reason = service.explain_anomaly(feature_vector)
    
    # Explanation should pinpoint roll_std_15
    assert "roll_std_15" in reason
    assert "20.0 std devs" in reason
