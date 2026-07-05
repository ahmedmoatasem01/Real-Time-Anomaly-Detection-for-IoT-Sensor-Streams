import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.database import Base, Reading, Alert, ModelRun

# We create an in-memory SQLite engine for tests
@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

def test_database_insert_reading(db_session):
    # Create reading
    reading = Reading(
        ts="2014-01-27T14:25:00",
        sensor_id="machine_temperature",
        value=72.13,
        anomaly_score=0.83,
        is_anomaly=True,
        severity="high",
        reason="high temperature",
        model="isolation_forest"
    )
    db_session.add(reading)
    db_session.commit()
    
    # Read back and verify
    fetched = db_session.query(Reading).first()
    assert fetched is not None
    assert fetched.value == 72.13
    assert fetched.is_anomaly is True
    assert fetched.severity == "high"

def test_database_alert_relation_and_ack(db_session):
    # Insert reading
    reading = Reading(
        ts="2014-01-27T14:25:00",
        sensor_id="machine_temperature",
        value=72.13,
        anomaly_score=0.83,
        is_anomaly=True,
        severity="high",
        reason="high temperature",
        model="isolation_forest"
    )
    db_session.add(reading)
    db_session.commit()
    
    # Create and link alert
    alert = Alert(
        reading_id=reading.id,
        ts=reading.ts,
        sensor_id=reading.sensor_id,
        severity=reading.severity,
        score=reading.anomaly_score,
        reason=reading.reason,
        acknowledged=False
    )
    db_session.add(alert)
    db_session.commit()
    
    # Fetch alert
    fetched_alert = db_session.query(Alert).first()
    assert fetched_alert is not None
    assert fetched_alert.reading_id == reading.id
    assert fetched_alert.acknowledged is False
    
    # Acknowledge
    fetched_alert.acknowledged = True
    db_session.commit()
    
    fetched_alert_updated = db_session.query(Alert).first()
    assert fetched_alert_updated.acknowledged is True

def test_database_model_run_insert(db_session):
    # Insert run
    run = ModelRun(
        model="isolation_forest",
        trained_at="2026-07-02T01:57:54",
        precision=0.92,
        recall=0.85,
        f1=0.88,
        pr_auc=0.90,
        roc_auc=0.95,
        threshold=0.52,
        notes="trained on raw temperature"
    )
    db_session.add(run)
    db_session.commit()
    
    fetched = db_session.query(ModelRun).first()
    assert fetched is not None
    assert fetched.f1 == 0.88
    assert fetched.model == "isolation_forest"
