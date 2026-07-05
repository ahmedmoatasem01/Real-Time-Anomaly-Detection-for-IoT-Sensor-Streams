import pytest
from src.drift.drift_detector import population_stability_index, mean_shift_sigma, classify, compute_drift_for_feature
from src.drift.drift_service import DriftService

def test_psi_identical():
    expected = [0.1] * 10
    actual = [0.1] * 10
    psi = population_stability_index(expected, actual)
    assert psi == pytest.approx(0.0, abs=1e-5)

def test_psi_shifted():
    expected = [0.1] * 10
    actual = [0.01] * 5 + [0.19] * 5
    psi = population_stability_index(expected, actual)
    assert psi > 0.1

def test_classify():
    assert classify(0.05, 1.0) == "stable"
    assert classify(0.15, 1.0) == "warning"
    assert classify(0.05, 2.5) == "warning"
    assert classify(0.3, 1.0) == "critical"
    assert classify(0.05, 3.5) == "critical"

def test_drift_service(monkeypatch):
    ds = DriftService()
    ds.baseline_stats = {"f1": {"mean": 0.0, "std": 1.0}}
    
    # Mock compute_drift_for_feature for stable state
    monkeypatch.setattr("src.drift.drift_service.compute_drift_for_feature", lambda *args: (0.05, 0.5))
    
    for _ in range(200):
        ds.update({"f1": 0.0})
    
    res = ds.check()
    assert res["status"] == "stable"
    
    # Mock compute_drift_for_feature for critical state
    monkeypatch.setattr("src.drift.drift_service.compute_drift_for_feature", lambda *args: (0.3, 5.0))
        
    res2 = ds.check()
    assert res2["status"] == "critical"
    assert res2["recommendation"] == "Retraining recommended"
