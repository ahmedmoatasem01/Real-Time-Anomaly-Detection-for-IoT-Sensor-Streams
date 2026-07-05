import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from src.streaming.stream_simulator import run_simulator

def test_stream_simulator_mock():
    # Mock data to simulate stream replayer
    data = {
        "timestamp": pd.date_range(start="2014-01-27 14:00:00", periods=5, freq="5min"),
        "sensor_id": ["machine_temperature"] * 5,
        "value": [72.1, 72.2, 72.3, 72.4, 72.5]
    }
    df = pd.DataFrame(data)
    
    mock_csv = "tests/mock_stream_data.csv"
    os.makedirs("tests", exist_ok=True)
    df.to_csv(mock_csv, index=False)
    
    # Mock requests.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "timestamp": "2014-01-27T14:00:00",
        "sensor_id": "machine_temperature",
        "value": 72.1,
        "anomaly_score": 0.1,
        "is_anomaly": False,
        "severity": "none",
        "reason": "Normal operation",
        "model": "isolation_forest",
        "inference_ms": 1.5
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        # Run simulator for 2 steps (without loop)
        # We patch time.sleep to run instantly
        with patch("time.sleep", return_value=None):
            run_simulator(
                csv_path=mock_csv,
                speed=1000.0,
                loop=False,
                start_index=0
            )
            
        # Verify requests.post was called for each row (5 rows)
        assert mock_post.call_count == 5
        
    # Clean up mock file
    if os.path.exists(mock_csv):
        os.remove(mock_csv)
