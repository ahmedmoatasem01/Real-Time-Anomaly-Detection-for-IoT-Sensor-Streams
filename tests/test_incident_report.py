import pytest
from src.reports.incident_report import generate_incident_pdf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.database import Base, get_db, Alert, Reading

# Setup test DB
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_alerts.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(autouse=True)
def setup_db(monkeypatch):
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    # Mock SessionLocal
    monkeypatch.setattr("src.reports.incident_report.SessionLocal", TestingSessionLocal)
    
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


def test_generate_incident_pdf():
    pdf_buffer = generate_incident_pdf(1)
    
    # Check if buffer has contents and is a PDF
    content = pdf_buffer.getvalue()
    assert len(content) > 100
    assert content.startswith(b"%PDF")
