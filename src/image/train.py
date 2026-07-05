import os
import json
import joblib
import datetime
from sklearn.ensemble import IsolationForest

from src.utils.logger import get_logger
from src.image.embeddings import get_extractor

logger = get_logger("image_train")

VISION_DIR = "data/raw/vision"
MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

def load_images_and_embed(directory: str):
    extractor = get_extractor()
    embeddings = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist.")
        return embeddings
        
    for f in os.listdir(directory):
        if f.endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(directory, f)
            emb = extractor.extract_embedding(path)
            embeddings.append(emb)
            
    return embeddings

def update_registry(metrics: dict):
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = []
        
    entry = {
        "name": "vision_iforest",
        "modality": "image",
        "dataset": "Synthetic Vision",
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "metrics": metrics,
        "is_production": True,
        "artifact_path": f"{MODELS_DIR}/vision_iforest.pkl"
    }
    
    updated = False
    for i, mod in enumerate(registry):
        if mod["name"] == "vision_iforest":
            registry[i] = entry
            updated = True
            break
            
    if not updated:
        registry.append(entry)
        
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)
    logger.info(f"Registered vision_iforest in {REGISTRY_PATH}")

def train_models():
    logger.info("Extracting embeddings for train/good...")
    train_embeddings = load_images_and_embed(os.path.join(VISION_DIR, "train", "good"))
    
    if not train_embeddings:
        logger.error("No training data found. Did you run the synthetic generator?")
        return
        
    logger.info(f"Training Isolation Forest on {len(train_embeddings)} images (512-dim)...")
    
    # Train Isolation Forest
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(train_embeddings)
    
    # Evaluate
    logger.info("Extracting embeddings for test/good...")
    test_good_emb = load_images_and_embed(os.path.join(VISION_DIR, "test", "good"))
    
    logger.info("Extracting embeddings for test/defective...")
    test_defect_emb = load_images_and_embed(os.path.join(VISION_DIR, "test", "defective"))
    
    if test_good_emb and test_defect_emb:
        good_scores = clf.decision_function(test_good_emb)
        defect_scores = clf.decision_function(test_defect_emb)
        
        logger.info(f"Test Good Scores (mean): {good_scores.mean():.4f}")
        logger.info(f"Test Defect Scores (mean): {defect_scores.mean():.4f}")
        
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "vision_iforest.pkl"))
    
    metrics = {
        "train_samples": len(train_embeddings),
        "good_mean_score": float(good_scores.mean()) if test_good_emb else 0.0,
        "defect_mean_score": float(defect_scores.mean()) if test_defect_emb else 0.0,
        "threshold": float((good_scores.mean() + defect_scores.mean()) / 2.0) if test_good_emb and test_defect_emb else 0.0
    }
    update_registry(metrics)
    logger.info("Vision training complete.")

if __name__ == "__main__":
    train_models()
