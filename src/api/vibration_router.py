from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
import os
import joblib
import json
import numpy as np

from src.vibration.data_loader import get_snapshot_files, load_vibration_snapshot
from src.vibration.features import extract_all_features
from scipy.fft import rfft, rfftfreq

router = APIRouter(prefix="/vibration", tags=["Vibration"])

VIBRATION_DIR = "data/raw/bearing/2nd_test"
MODELS_DIR = "models"

# Lazy load artifacts
scaler = None
model = None
feature_cols = None
files = []

def load_artifacts():
    global scaler, model, feature_cols, files
    if not files:
        files = get_snapshot_files(VIBRATION_DIR)
    
    if scaler is None:
        scaler_path = os.path.join(MODELS_DIR, "vibration_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
    if model is None:
        model_path = os.path.join(MODELS_DIR, "vibration_iforest.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
    if feature_cols is None:
        cols_path = os.path.join(MODELS_DIR, "vibration_feature_columns.json")
        if os.path.exists(cols_path):
            with open(cols_path, "r") as f:
                feature_cols = json.load(f)

@router.get("/status")
def get_status():
    load_artifacts()
    return {
        "dataset_available": len(files) > 0,
        "total_snapshots": len(files),
        "model_loaded": model is not None
    }

@router.get("/sample")
def get_vibration_sample(file_index: int = Query(0, description="Index of the snapshot in chronological order")):
    load_artifacts()
    
    if not files:
        raise HTTPException(status_code=404, detail="Vibration dataset not found")
        
    if file_index < 0 or file_index >= len(files):
        raise HTTPException(status_code=400, detail=f"Invalid file_index. Must be between 0 and {len(files)-1}")
        
    file_path = files[file_index]
    signal = load_vibration_snapshot(file_path, bearing_idx=0)
    
    if len(signal) == 0:
        raise HTTPException(status_code=500, detail="Failed to load signal")
        
    # Extract features
    features = extract_all_features(signal)
    
    # Run inference
    score = 0.0
    is_anomaly = False
    
    if model and scaler and feature_cols:
        # Create input array in correct order
        input_vector = [features.get(c, 0.0) for c in feature_cols]
        X = scaler.transform([input_vector])
        # decision_function: lower means more anomalous. Invert it so higher = anomalous
        raw_score = model.decision_function(X)[0]
        score = -float(raw_score) 
        
        # Simple threshold based on isolation forest (e.g., score > 0 is anomalous)
        is_anomaly = bool(score > 0)
    
    # Downsample waveform for UI rendering (e.g. 500 points max)
    downsample_factor = max(1, len(signal) // 500)
    ui_waveform = signal[::downsample_factor].tolist()
    
    # Compute FFT for UI
    yf = rfft(signal)
    xf = rfftfreq(len(signal), 1 / 20000.0)
    
    # Take first 200 bins of FFT for visualization
    fft_magnitudes = np.abs(yf)[:200].tolist()
    fft_freqs = xf[:200].tolist()
    
    return {
        "file_index": file_index,
        "timestamp": os.path.basename(file_path),
        "features": features,
        "anomaly_score": score,
        "is_anomaly": is_anomaly,
        "waveform": ui_waveform,
        "fft": {
            "frequencies": fft_freqs,
            "magnitudes": fft_magnitudes
        }
    }

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

@router.websocket("/ws/stream")
async def websocket_vibration_stream(websocket: WebSocket):
    await websocket.accept()
    load_artifacts()
    
    if not files:
        await websocket.close(code=1011, reason="No vibration data available")
        return

    current_index = 0
    is_playing = True
    
    # We need to run a background task that streams data if is_playing is True
    # while listening to the websocket for control messages (play/pause).
    
    async def stream_data():
        nonlocal current_index, is_playing
        while True:
            if is_playing and current_index < len(files):
                # We can reuse the logic from get_vibration_sample but async
                try:
                    # To avoid blocking the event loop, we should technically use run_in_threadpool
                    # but for simplicity we will just call it synchronously here since it's fast enough
                    # for a 1-second interval.
                    from fastapi.concurrency import run_in_threadpool
                    
                    data = await run_in_threadpool(get_vibration_sample, current_index)
                    await websocket.send_json(data)
                    current_index += 1
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    is_playing = False
                    
            await asyncio.sleep(1.0) # 1 second per snapshot

    stream_task = asyncio.create_task(stream_data())
    
    try:
        while True:
            msg = await websocket.receive_json()
            command = msg.get("command")
            if command == "play":
                is_playing = True
            elif command == "pause":
                is_playing = False
            elif command == "reset":
                current_index = 0
                is_playing = False
            elif command == "seek":
                idx = msg.get("index")
                if idx is not None and 0 <= idx < len(files):
                    current_index = idx
    except WebSocketDisconnect:
        print("Vibration WS client disconnected")
    finally:
        stream_task.cancel()
