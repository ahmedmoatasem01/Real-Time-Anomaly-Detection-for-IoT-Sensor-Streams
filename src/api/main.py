import datetime
import json
from contextlib import asynccontextmanager
from typing import List, Optional, Set, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.database.database import init_db, Reading, Alert, ModelRun, get_db
from src.api.schemas import ReadingCreate, PredictionResponse, AlertResponse, AlertStatusUpdate, AlertFeedback, FaultInjectRequest, FaultStatus
from src.api.inference_service import InferenceService
from src.synthetic.synthetic_stream import fault_manager
from src.synthetic.fault_generator import FAULT_TYPES
from src.api.vibration_router import router as vibration_router
from src.api.image_router import router as image_router
from src.api.asset_router import router as asset_router

logger = get_logger("api")
settings = get_settings()

# ── Lifespan handler (replaces deprecated @app.on_event) ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized via lifespan startup.")
    yield
    logger.info("API shutting down.")

app = FastAPI(
    title="Real-Time Anomaly Detection API",
    version="1.0.0",
    description="IoT sensor stream anomaly detection using Isolation Forest",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vibration_router)
app.include_router(image_router)
app.include_router(asset_router)

# ── Inference service singleton ───────────────────────────────────────────────
inference_service = InferenceService()


# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WS client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WS client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        import asyncio
        dead = set()
        for connection in list(self.active_connections):
            try:
                # Use wait_for to prevent a stuck connection from blocking everything
                await asyncio.wait_for(connection.send_json(message), timeout=0.5)
            except Exception as e:
                logger.warning(f"WS broadcast error: {e}")
                dead.add(connection)
        self.active_connections -= dead

manager = ConnectionManager()


# ── Helper: build DB Reading object ──────────────────────────────────────────
def _make_db_reading(pred: dict) -> Reading:
    return Reading(
        ts=pred["timestamp"],
        sensor_id=pred["sensor_id"],
        value=pred["value"],
        anomaly_score=pred["anomaly_score"],
        is_anomaly=pred["is_anomaly"],
        severity=pred["severity"],
        reason=pred["reason"],
        model=pred["model"],
        inference_ms=pred.get("inference_ms", 0.0),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "isolation_forest" if inference_service.model is not None else "none",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.get("/faults/types", tags=["System"])
def get_fault_types():
    return {"types": FAULT_TYPES, "descriptions": {}}

@app.post("/faults/inject", response_model=FaultStatus, tags=["System"])
def inject_fault(req: FaultInjectRequest):
    """Injects a synthetic fault into the live stream."""
    if req.fault_type == "vibration_fault":
        raise HTTPException(status_code=422, detail="vibration_fault is reserved for the future vibration module")
    return fault_manager.start_fault(
        fault_type=req.fault_type,
        duration_steps=req.duration_steps,
        magnitude=req.magnitude,
        sensor_id=req.sensor_id
    )

@app.post("/faults/stop", response_model=FaultStatus, tags=["System"])
def stop_fault():
    """Stops any currently active synthetic fault."""
    return fault_manager.stop_fault()

@app.get("/faults/status", response_model=FaultStatus, tags=["System"])
def get_fault_status():
    """Returns the current synthetic fault status."""
    return fault_manager.get_status()

from fastapi.concurrency import run_in_threadpool

@app.post("/predict/ensemble", response_model=Dict[str, Any], tags=["Inference"])
async def predict_ensemble(payload: ReadingCreate):
    """
    Scores the reading across all available models to provide a consensus anomaly decision.
    """
    try:
        def _do_ensemble():
            return inference_service.ensemble_predict(
                timestamp=payload.timestamp,
                sensor_id=payload.sensor_id,
                value=payload.value
            )
        res = await run_in_threadpool(_do_ensemble)
        return res
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from src.drift.drift_service import drift_service

@app.get("/drift/status", response_model=Dict[str, Any], tags=["System"])
async def get_drift_status():
    """
    Returns the current statistical drift status of the data stream.
    """
    try:
        return drift_service.get_status()
    except Exception as e:
        logger.error(f"Drift status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/check", response_model=Dict[str, Any], tags=["System"])
async def force_drift_check():
    """
    Forces an immediate computation of drift over the current live window.
    """
    try:
        return drift_service.check()
    except Exception as e:
        logger.error(f"Drift check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/history", response_model=List[Dict[str, Any]], tags=["System"])
async def get_drift_history():
    """
    Returns recent drift check history.
    """
    import os
    import json
    history_path = "reports/drift_history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading drift history: {e}")
    return []


@app.post("/predict", response_model=PredictionResponse)
async def predict(reading: ReadingCreate, db: Session = Depends(get_db)):
    def _do_predict():
        # 1. Run inference
        pred = inference_service.predict(
            timestamp=reading.timestamp,
            sensor_id=reading.sensor_id,
            value=reading.value,
        )

        # 2. Persist reading
        db_reading = _make_db_reading(pred)
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)

        # 3. Persist alert if anomalous
        db_alert_id = None
        if pred["is_anomaly"]:
            db_alert = Alert(
                reading_id=db_reading.id,
                ts=pred["timestamp"],
                sensor_id=pred["sensor_id"],
                severity=pred["severity"],
                score=pred["anomaly_score"],
                reason=pred["reason"],
                acknowledged=False,
            )
            db.add(db_alert)
            db.commit()
            db.refresh(db_alert)
            db_alert_id = db_alert.id
            
        return pred, db_reading.id, db_alert_id

    pred, db_reading_id, db_alert_id = await run_in_threadpool(_do_predict)

    # 4. Broadcast over WebSocket
    broadcast_payload = dict(pred)
    broadcast_payload["reading_id"] = db_reading_id
    if db_alert_id is not None:
        broadcast_payload["alert_id"] = db_alert_id

    await manager.broadcast(broadcast_payload)

    return pred


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(readings: List[ReadingCreate], db: Session = Depends(get_db)):
    def _do_batch():
        results = []
        for r in readings:
            pred = inference_service.predict(
                timestamp=r.timestamp,
                sensor_id=r.sensor_id,
                value=r.value,
            )
            db_reading = _make_db_reading(pred)
            db.add(db_reading)
            db.commit()
            db.refresh(db_reading)

            if pred["is_anomaly"]:
                db_alert = Alert(
                    reading_id=db_reading.id,
                    ts=pred["timestamp"],
                    sensor_id=pred["sensor_id"],
                    severity=pred["severity"],
                    score=pred["anomaly_score"],
                    reason=pred["reason"],
                    acknowledged=False,
                )
                db.add(db_alert)
                db.commit()

            results.append(pred)
        return results

    results = await run_in_threadpool(_do_batch)
    return results


@app.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    limit: int = 50,
    acknowledged: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Alert)
    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)
    return query.order_by(Alert.ts.desc()).limit(limit).all()


@app.post("/alerts/{alert_id}/ack")
def ack_alert(alert_id: int, db: Session = Depends(get_db)):
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not db_alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    db_alert.acknowledged = True  # type: ignore[assignment]
    db.commit()
    return {"status": "success", "message": f"Alert {alert_id} acknowledged"}


@app.put("/alerts/{alert_id}/status")
def update_alert_status(alert_id: int, req: AlertStatusUpdate, db: Session = Depends(get_db)):
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not db_alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    db_alert.status = req.status # type: ignore[assignment]
    if req.operator_note is not None:
        db_alert.operator_note = req.operator_note # type: ignore[assignment]
        
    if req.status == "resolved" or req.status == "false_alarm":
        db_alert.resolved_at = datetime.datetime.utcnow() # type: ignore[assignment]
        
    db.commit()
    return {"status": "success", "message": f"Alert {alert_id} updated to {req.status}"}


@app.put("/alerts/{alert_id}/feedback")
def update_alert_feedback(alert_id: int, req: AlertFeedback, db: Session = Depends(get_db)):
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not db_alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    db_alert.feedback = req.feedback # type: ignore[assignment]
    db.commit()
    return {"status": "success", "message": f"Alert {alert_id} feedback set to {req.feedback}"}


@app.get("/alerts/{alert_id}/replay", response_model=List[PredictionResponse])
def replay_alert(alert_id: int, context_steps: int = 30, db: Session = Depends(get_db)):
    """Replays the readings surrounding an alert."""
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not db_alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    # Get the central reading
    reading_id = db_alert.reading_id
    
    # Query surrounding readings based on ID (assuming sequential insertions)
    start_id = max(1, reading_id - context_steps)
    end_id = reading_id + context_steps
    
    readings = db.query(Reading).filter(Reading.id >= start_id, Reading.id <= end_id).all()
    
    return [
        {
            "timestamp": r.ts,
            "sensor_id": r.sensor_id,
            "value": r.value,
            "anomaly_score": r.anomaly_score,
            "is_anomaly": r.is_anomaly,
            "severity": r.severity,
            "reason": r.reason,
            "model": r.model,
            "inference_ms": r.inference_ms,
        }
        for r in readings
    ]


from fastapi.responses import StreamingResponse

@app.get("/reports/incident/{alert_id}")
def get_incident_report(alert_id: int):
    """Generates and returns an incident report PDF."""
    from src.reports.incident_report import generate_incident_pdf
    try:
        pdf_buffer = generate_incident_pdf(alert_id)
        return StreamingResponse(
            pdf_buffer, 
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename=incident_{alert_id}.pdf"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not generate report")


@app.get("/metrics")
def get_system_metrics(db: Session = Depends(get_db)):
    total_readings = db.query(Reading).count()
    total_anomalies = db.query(Reading).filter(Reading.is_anomaly == True).count()

    # Average inference latency from persisted values
    avg_latency_row = db.query(func.avg(Reading.inference_ms)).scalar()
    avg_latency_val = float(avg_latency_row) if avg_latency_row is not None else 0.0

    # Stream rate: readings in last 60 seconds
    one_min_ago = (datetime.datetime.utcnow() - datetime.timedelta(seconds=60)).isoformat()
    recent_count = db.query(Reading).filter(Reading.ts >= one_min_ago).count()

    latest_run = db.query(ModelRun).order_by(ModelRun.id.desc()).first()

    return {
        "total_readings": total_readings,
        "total_anomalies": total_anomalies,
        "anomaly_rate": round(total_anomalies / max(total_readings, 1) * 100, 2),
        "avg_inference_latency_ms": round(avg_latency_val, 3),
        "current_model": "isolation_forest" if inference_service.model is not None else "none",
        "stream_rate_rpm": recent_count,
        "latest_run": {
            "precision": round(float(latest_run.precision), 4),  # type: ignore[arg-type]
            "recall": round(float(latest_run.recall), 4),  # type: ignore[arg-type]
            "f1": round(float(latest_run.f1), 4),  # type: ignore[arg-type]
            "pr_auc": round(float(latest_run.pr_auc), 4),  # type: ignore[arg-type]
            "roc_auc": round(float(latest_run.roc_auc), 4),  # type: ignore[arg-type]
            "threshold": round(float(latest_run.threshold), 6),  # type: ignore[arg-type]
        } if latest_run else None,
    }


@app.get("/readings")
def get_recent_readings(limit: int = 60, db: Session = Depends(get_db)):
    """Return the most recent N readings for frontend chart initialisation."""
    readings = db.query(Reading).order_by(Reading.ts.desc()).limit(limit).all()
    return [
        {
            "timestamp": r.ts,
            "sensor_id": r.sensor_id,
            "value": r.value,
            "anomaly_score": r.anomaly_score,
            "is_anomaly": r.is_anomaly,
            "severity": r.severity,
            "reason": r.reason,
            "model": r.model,
            "inference_ms": r.inference_ms,
        }
        for r in reversed(readings)
    ]


# ── Task 2.8 — Model registry / comparison / experiment endpoints ─────────────

@app.get("/models")
def get_models():
    """List all registered models with basic metrics."""
    try:
        from src.registry.model_registry import list_models
        return list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/registry")
def get_model_registry():
    """Return the full model registry JSON."""
    import os
    path = os.path.join("models", "model_registry.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@app.get("/models/comparison")
def get_model_comparison():
    """Return ranked model comparison JSON for the frontend."""
    import os
    path = "reports/model_comparison.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="model_comparison.json not found. Run evaluate_all.py first.")
    with open(path) as f:
        return json.load(f)


@app.get("/experiments")
def get_experiments():
    """Return full experiment history."""
    try:
        from src.experiments.experiment_tracker import load_experiments
        return load_experiments()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/summary")
def get_data_summary():
    """Dataset metadata for the Data Explorer page."""
    import os, csv
    raw_path = settings.RAW_CSV
    proc_path = settings.PROCESSED_CSV
    feat_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")

    summary = {
        "dataset": "NAB — machine_temperature_system_failure",
        "source": "realKnownCause/machine_temperature_system_failure.csv",
        "raw_exists": os.path.exists(raw_path),
        "processed_exists": os.path.exists(proc_path),
    }
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            summary["feature_columns"] = json.load(f)
        summary["feature_count"] = len(summary["feature_columns"])

    if os.path.exists(proc_path):
        df_meta = {}
        with open(proc_path, newline="") as cf:
            reader = csv.DictReader(cf)
            rows = list(reader)
        df_meta["total_rows"] = len(rows)
        splits = {}
        labels = {"0": 0, "1": 0}
        for r in rows:
            s = r.get("split", "unknown")
            splits[s] = splits.get(s, 0) + 1
            lbl = r.get("label", "0")
            labels[lbl] = labels.get(lbl, 0) + 1
        df_meta["splits"] = splits
        df_meta["label_counts"] = {"normal": labels.get("0", 0), "anomaly": labels.get("1", 0)}
        df_meta["anomaly_rate_pct"] = round(labels.get("1", 0) / max(len(rows), 1) * 100, 2)
        summary.update(df_meta)

    return summary


@app.get("/system/status")
def get_system_status(db: Session = Depends(get_db)):
    """API / DB / WS / model / stream health for the System Health page."""
    db_ok = False
    try:
        db.execute(func.now() if False else __import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    total_readings = db.query(Reading).count()
    one_min_ago    = (datetime.datetime.utcnow() - datetime.timedelta(seconds=60)).isoformat()
    recent_count   = db.query(Reading).filter(Reading.ts >= one_min_ago).count()

    avg_latency_row = db.query(func.avg(Reading.inference_ms)).scalar()
    avg_latency     = round(float(avg_latency_row), 3) if avg_latency_row else 0.0

    return {
        "api":            "ok",
        "database":       "ok" if db_ok else "error",
        "websocket_clients": len(manager.active_connections),
        "production_model":  inference_service.model is not None and "isolation_forest" or "none",
        "threshold":         round(inference_service.threshold, 6),
        "total_readings":    total_readings,
        "stream_rate_rpm":   recent_count,
        "avg_inference_ms":  avg_latency,
        "timestamp":         datetime.datetime.utcnow().isoformat(),
    }


@app.post("/models/select/{name}")
def select_model(name: str):
    """
    Hot-swap the production model in the live InferenceService.
    Updates is_production in model_registry.json.
    """
    import os, joblib
    from src.registry.model_registry import get as reg_get, set_production

    entry = reg_get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not in registry.")

    artifact_path = entry.get("artifact_path")
    threshold_path = entry.get("threshold_path")

    if name == "rolling_zscore":
        raise HTTPException(status_code=400, detail="Rolling Z-score baseline cannot be set as live model.")

    if not artifact_path or not os.path.exists(artifact_path):
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")

    # Reload model + threshold into the running InferenceService
    inference_service.model = joblib.load(artifact_path)
    if "metrics" in entry and "threshold" in entry["metrics"]:
        inference_service.threshold = entry["metrics"]["threshold"]
    elif threshold_path and os.path.exists(threshold_path):
        with open(threshold_path) as f:
            inference_service.threshold = json.load(f)["threshold"]

    set_production(name)
    logger.info(f"Hot-swapped production model to '{name}' (threshold={inference_service.threshold:.6f})")
    return {"status": "ok", "active_model": name, "threshold": inference_service.threshold}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; data is pushed by /predict
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WS connection error: {e}")
        manager.disconnect(websocket)

@app.post("/models/retrain")
def trigger_retraining():
    from src.retraining.retrain_pipeline import run_retraining
    import threading
    
    def background_retrain():
        try:
            run_retraining()
        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)
            
    # Run in background to not block the request
    t = threading.Thread(target=background_retrain)
    t.start()
    
    return {"status": "success", "message": "Retraining pipeline triggered in background"}

