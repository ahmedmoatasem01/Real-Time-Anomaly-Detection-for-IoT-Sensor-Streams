import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.api.main import app
from src.database.database import Base, get_db, Alert, Reading
import datetime

# Setup test DB
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_alerts.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    # Insert dummy reading and alert
    r = Reading(ts="2024-01-01T00:00:00", sensor_id="s1", value=10.0, anomaly_score=0.9, is_anomaly=True, severity="high", reason="test", model="test")
    db.add(r)
    db.commit()
    
    a = Alert(reading_id=r.id, ts=r.ts, sensor_id=r.sensor_id, severity=r.severity, score=r.anomaly_score, reason=r.reason, acknowledged=False)
    db.add(a)
    db.commit()
    
    yield
    
    db.close()
    Base.metadata.drop_all(bind=engine)

def test_update_alert_status():
    res = client.put("/alerts/1/status", json={"status": "investigating", "operator_note": "looking into this"})
    assert res.status_code == 200
    
    # verify
    db = TestingSessionLocal()
    a = db.query(Alert).filter(Alert.id == 1).first()
    assert a.status == "investigating"
    assert a.operator_note == "looking into this"
    assert a.resolved_at is None
    db.close()

def test_update_alert_feedback():
    res = client.put("/alerts/1/feedback", json={"feedback": "true_positive"})
    assert res.status_code == 200
    
    # verify
    db = TestingSessionLocal()
    a = db.query(Alert).filter(Alert.id == 1).first()
    assert a.feedback == "true_positive"
    db.close()

def test_resolve_alert():
    res = client.put("/alerts/1/status", json={"status": "resolved"})
    assert res.status_code == 200
    
    db = TestingSessionLocal()
    a = db.query(Alert).filter(Alert.id == 1).first()
    assert a.status == "resolved"
    assert a.resolved_at is not None
    db.close()
