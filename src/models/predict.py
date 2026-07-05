import pandas as pd
from typing import Dict, Any
from src.api.inference_service import InferenceService

class Predictor:
    """
    Standalone helper class to score a single reading or batch of readings.
    Wraps InferenceService for non-API usages.
    """
    def __init__(self):
        self.service = InferenceService()
        
    def predict_one(self, timestamp: str, sensor_id: str, value: float) -> Dict[str, Any]:
        """
        Score a single reading.
        """
        return self.service.predict(timestamp, sensor_id, value)
        
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch of readings (DataFrame format).
        Expected columns: timestamp, sensor_id, value
        """
        results = []
        for _, row in df.iterrows():
            res = self.predict_one(
                timestamp=str(row["timestamp"]),
                sensor_id=str(row["sensor_id"]),
                value=float(row["value"])
            )
            results.append(res)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Quick test
    import datetime
    predictor = Predictor()
    now = datetime.datetime.utcnow().isoformat()
    res = predictor.predict_one(now, "machine_temperature", 72.5)
    print("Prediction Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
