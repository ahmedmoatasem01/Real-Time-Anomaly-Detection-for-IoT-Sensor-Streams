import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.database import Base, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Create test DB engine and session
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Override get_db dependency in FastAPI app
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module", autouse=True)
def init_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

def test_predict_endpoint():
    with TestClient(app) as client:
        payload = {
            "timestamp": "2014-01-27T14:25:00",
            "sensor_id": "machine_temperature",
            "value": 72.13
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["timestamp"] == "2014-01-27T14:25:00"
        assert data["sensor_id"] == "machine_temperature"
        assert data["value"] == 72.13
        assert "is_anomaly" in data
