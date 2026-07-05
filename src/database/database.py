import os
import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("database")
settings = get_settings()

Base = declarative_base()

class Reading(Base):
    __tablename__ = "readings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False, index=True)
    sensor_id = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=True)
    is_anomaly = Column(Boolean, nullable=True)
    severity = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    model = Column(String, nullable=True)
    inference_ms = Column(Float, nullable=True, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    alerts = relationship("Alert", back_populates="reading", cascade="all, delete-orphan")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    reading_id = Column(Integer, ForeignKey("readings.id", ondelete="CASCADE"), nullable=False)
    ts = Column(String, nullable=False)
    sensor_id = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    reason = Column(String, nullable=True)
    acknowledged = Column(Boolean, default=False)
    status = Column(String, default="new")
    operator_note = Column(String, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    feedback = Column(String, nullable=True)
    
    reading = relationship("Reading", back_populates="alerts")

class ModelRun(Base):
    __tablename__ = "model_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String, nullable=False)
    trained_at = Column(String, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1 = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    notes = Column(String, nullable=True)

class Asset(Base):
    __tablename__ = "assets"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    modality = Column(String, nullable=False)
    status = Column(String, default="operational")
    last_anomaly_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def get_db_engine():
    db_url = settings.DB_URL
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        if db_path:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    engine = create_engine(
        db_url, 
        connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {}
    )
    return engine

engine = get_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    logger.info(f"Initializing database at {settings.DB_URL}...")
    Base.metadata.create_all(bind=engine)
    
    # Pre-seed the assets if they don't exist
    with get_db_session() as db:
        if db.query(Asset).count() == 0:
            assets = [
                Asset(id="Machine-01", name="Rotary Compressor A", modality="time_series", status="operational"),
                Asset(id="Bearing-01", name="NASA Bearing Set 2", modality="vibration", status="warning"),
                Asset(id="Product-Inspection-01", name="Quality Cam 1", modality="vision", status="operational")
            ]
            db.add_all(assets)
            logger.info("Pre-seeded 3 assets into the database.")
            
    logger.info("Database initialized successfully.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
