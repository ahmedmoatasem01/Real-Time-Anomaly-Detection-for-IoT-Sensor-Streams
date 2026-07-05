from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database.database import get_db, Asset, Reading, Alert, ModelRun
import json
import os

router = APIRouter(prefix="/assets", tags=["Assets"])

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

@router.get("/")
def get_assets(db: Session = Depends(get_db)):
    assets = db.query(Asset).all()
    return {
        "assets": [
            {
                "id": a.id,
                "name": a.name,
                "modality": a.modality,
                "status": a.status,
                "last_anomaly_at": a.last_anomaly_at.isoformat() if a.last_anomaly_at else None
            } for a in assets
        ]
    }

@router.get("/summary")
def get_asset_summary(db: Session = Depends(get_db)):
    total_assets = db.query(Asset).count()
    
    # Read models from registry
    total_models = 0
    modalities = set()
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
            total_models = len(registry)
            for m in registry:
                modalities.add(m.get("modality", "time_series"))
                
    total_readings = db.query(Reading).count()
    total_alerts = db.query(Alert).count()
    
    return {
        "total_assets": total_assets,
        "active_modalities": len(modalities),
        "total_models": total_models,
        "total_readings_processed": total_readings,
        "total_anomalies_detected": total_alerts
    }
