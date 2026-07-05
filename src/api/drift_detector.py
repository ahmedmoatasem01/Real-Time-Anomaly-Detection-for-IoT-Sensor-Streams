import time
from river import drift
from src.utils.logger import get_logger

logger = get_logger("drift_detector")

class DriftDetector:
    def __init__(self):
        # We track drift on raw values and anomaly scores
        self.value_adwin = drift.ADWIN(delta=0.002)
        self.score_adwin = drift.ADWIN(delta=0.002)
        
        self.drift_status = "stable"
        self.last_drift_time = None
        self.drift_count = 0
        
    def update(self, value: float, score: float):
        """
        Updates the drift detectors with a new raw value and anomaly score.
        """
        self.value_adwin.update(value)
        self.score_adwin.update(score)
        
        value_drift = self.value_adwin.drift_detected
        score_drift = self.score_adwin.drift_detected
        
        if value_drift or score_drift:
            self.drift_status = "warning: retraining recommended"
            self.last_drift_time = time.time()
            self.drift_count += 1
            
            reasons = []
            if value_drift:
                reasons.append("Raw value drift detected")
            if score_drift:
                reasons.append("Anomaly score distribution shifted")
                
            logger.warning(f"Drift Detected! {' | '.join(reasons)}")
            
        # Optional: auto-recover status if no drift for a long time
        # For simplicity, we just keep it in warning state once drifted, 
        # or we could let the user reset it.

    def get_status(self) -> dict:
        return {
            "status": self.drift_status,
            "drift_count": self.drift_count,
            "last_drift_timestamp": self.last_drift_time
        }
