from fastapi import APIRouter, HTTPException, File, UploadFile
import os
import base64
import joblib
import random
import json
from pydantic import BaseModel

from src.image.embeddings import get_extractor

router = APIRouter(prefix="/image", tags=["Vision"])

VISION_DIR = "data/raw/vision"
MODELS_DIR = "models"
model = None

def load_model():
    global model
    if model is None:
        path = os.path.join(MODELS_DIR, "vision_iforest.pkl")
        if os.path.exists(path):
            model = joblib.load(path)

class AnalyzeRequest(BaseModel):
    image_base64: str

@router.get("/gallery")
def get_gallery():
    """Returns base64 encoded test images to populate the UI gallery"""
    gallery = []
    
    # Load some good and some defective
    test_dirs = [
        ("Good", os.path.join(VISION_DIR, "test", "good")),
        ("Defective", os.path.join(VISION_DIR, "test", "defective"))
    ]
    
    for label, d in test_dirs:
        if not os.path.exists(d):
            continue
        all_files = [f for f in os.listdir(d) if f.endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(all_files)
        for f in all_files[:5]: # Just return 5 of each
            path = os.path.join(d, f)
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = "image/jpeg" if f.endswith((".jpg", ".jpeg")) else "image/png"
                gallery.append({
                    "filename": f,
                    "label": label,
                    "data": f"data:{mime_type};base64,{encoded}"
                })
    random.shuffle(gallery)
    return {"images": gallery}

@router.post("/analyze")
def analyze_image(req: AnalyzeRequest):
    load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Vision model not trained yet")
        
    try:
        # Extract base64
        header, encoded = req.image_base64.split(",", 1) if "," in req.image_base64 else ("", req.image_base64)
        image_bytes = base64.b64decode(encoded)
        
        # Extract embedding
        extractor = get_extractor()
        emb = extractor.extract_embedding(image_bytes)
        
        # Inference
        raw_score = model.decision_function([emb])[0]
        
        # Load threshold from registry
        threshold = 0.0
        registry_path = os.path.join(MODELS_DIR, "model_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)
                for entry in registry:
                    if entry.get("name") == "vision_iforest":
                        threshold = entry.get("metrics", {}).get("threshold", 0.0)
                        break
                        
        score = -float(raw_score) # Keep for UI display convention
        
        is_anomaly = bool(raw_score < threshold)
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "message": "Defect detected" if is_anomaly else "Normal"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
