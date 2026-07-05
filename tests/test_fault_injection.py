import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.synthetic.fault_generator import FaultGenerator

client = TestClient(app)

def test_fault_generator_spike():
    fg = FaultGenerator("spike", 10, 50.0, "s1")
    v1 = fg.next_value(10.0)
    assert v1 == 60.0
    v2 = fg.next_value(10.0)
    assert v2 == 60.0
    v3 = fg.next_value(10.0)
    assert v3 == 60.0
    v4 = fg.next_value(10.0)
    assert v4 == 10.0
    assert not fg.finished

def test_fault_generator_drift():
    fg = FaultGenerator("gradual_drift", 10, 100.0, "s1")
    v1 = fg.next_value(0.0)
    assert v1 == 0.0
    v2 = fg.next_value(0.0)
    assert v2 == 10.0
    for _ in range(8):
        fg.next_value(0.0)
    assert fg.finished

def test_fault_generator_stuck():
    fg = FaultGenerator("sensor_stuck", 5, 0.0, "s1")
    v1 = fg.next_value(15.0)
    assert v1 == 15.0
    v2 = fg.next_value(20.0)
    assert v2 == 15.0

def test_fault_generator_missing():
    fg = FaultGenerator("missing_values", 5, 0.0, "s1")
    v1 = fg.next_value(10.0)
    assert v1 is None

def test_fault_generator_noise():
    fg = FaultGenerator("noise_burst", 5, 2.0, "s1")
    v1 = fg.next_value(10.0)
    assert v1 != 10.0

def test_fault_generator_overheating():
    fg = FaultGenerator("overheating", 10, 100.0, "s1")
    v1 = fg.next_value(0.0)
    assert v1 == 0.0
    v2 = fg.next_value(0.0)
    assert v2 == pytest.approx(1.0)
    v3 = fg.next_value(0.0)
    assert v3 == pytest.approx(4.0)

def test_api_fault_injection():
    # Inject
    res = client.post("/faults/inject", json={
        "fault_type": "gradual_drift",
        "duration_steps": 10,
        "magnitude": 5.0,
        "sensor_id": "machine_temperature"
    })
    assert res.status_code == 200
    assert res.json()["active"] is True
    assert res.json()["fault_type"] == "gradual_drift"
    
    # Status
    res = client.get("/faults/status")
    assert res.status_code == 200
    assert res.json()["active"] is True
    
    # Stop
    res = client.post("/faults/stop")
    assert res.status_code == 200
    assert res.json()["active"] is False
    
    # Status again
    res = client.get("/faults/status")
    assert res.status_code == 200
    assert res.json()["active"] is False

def test_api_vibration_fault_rejected():
    res = client.post("/faults/inject", json={
        "fault_type": "vibration_fault",
        "duration_steps": 10
    })
    assert res.status_code == 422
