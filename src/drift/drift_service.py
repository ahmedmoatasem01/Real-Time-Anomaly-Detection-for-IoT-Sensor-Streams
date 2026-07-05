import json
import os
import datetime
from collections import deque
from src.utils.logger import get_logger
from src.drift.drift_detector import compute_drift_for_feature, classify
from src.utils.config import get_settings

logger = get_logger("drift_service")
settings = get_settings()

class DriftService:
    def __init__(self):
        self.window = deque(maxlen=500)
        self.baseline_stats = {}
        self.load_baseline()

    def load_baseline(self):
        path = os.path.join(settings.MODEL_DIR, "feature_stats.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.baseline_stats = json.load(f)
            logger.info("Loaded baseline feature stats for drift detection.")

    def update(self, features: dict):
        self.window.append(features)

    def check(self) -> dict:
        if not self.baseline_stats:
            return {"status": "stable", "recommendation": "No baseline stats available"}

        if len(self.window) < 50:
            return {"status": "stable", "recommendation": "Waiting for more data"}

        max_psi = 0.0
        max_shift = 0.0
        worst_status = "stable"
        affected_features = []
        
        # We only check a subset of features or all of them. Let's check all available in stats.
        for feature, stats in self.baseline_stats.items():
            baseline_mean = stats["mean"]
            baseline_std = stats["std"]
            
            live_vals = [w.get(feature, 0.0) for w in self.window if feature in w]
            if not live_vals:
                continue
                
            psi, shift = compute_drift_for_feature(live_vals, baseline_mean, baseline_std)
            
            status = classify(psi, shift)
            
            if status != "stable":
                affected_features.append(feature)
                
            if status == "critical":
                worst_status = "critical"
            elif status == "warning" and worst_status == "stable":
                worst_status = "warning"
                
            if psi > max_psi:
                max_psi = psi
            if abs(shift) > abs(max_shift):
                max_shift = shift

        recommendation = "No action"
        if worst_status == "warning":
            recommendation = "Monitor"
        elif worst_status == "critical":
            recommendation = "Retraining recommended"

        # Overall live mean/baseline mean doesn't make total sense across all features, 
        # so we will report the max shift. The UI just wants one shift number usually.
        # Let's return the metrics of the most shifted feature.
        
        result = {
            "baseline_mean": 0.0, # placeholder, could be specific feature
            "live_mean": 0.0,
            "mean_shift_sigma": max_shift,
            "psi": max_psi,
            "affected_features": affected_features,
            "status": worst_status,
            "recommendation": recommendation,
            "checked_at": datetime.datetime.utcnow().isoformat()
        }

        # Persist
        try:
            os.makedirs("reports", exist_ok=True)
            with open("reports/drift_status.json", "w") as f:
                json.dump(result, f)
                
            history_path = "reports/drift_history.json"
            history = []
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
            history.append({
                "timestamp": result["checked_at"],
                "status": result["status"],
                "psi": result["psi"]
            })
            # keep last 100
            history = history[-100:]
            with open(history_path, "w") as f:
                json.dump(history, f)
        except Exception as e:
            logger.error(f"Error persisting drift: {e}")

        return result

    def get_status(self) -> dict:
        path = "reports/drift_status.json"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"status": "stable", "mean_shift_sigma": 0.0, "psi": 0.0, "affected_features": [], "recommendation": "No action", "checked_at": datetime.datetime.utcnow().isoformat()}

drift_service = DriftService()
