from pydantic import BaseModel, ConfigDict
from typing import Optional, List


class ReadingCreate(BaseModel):
    timestamp: str
    sensor_id: str
    value: float


class PredictionResponse(BaseModel):
    timestamp: str
    sensor_id: str
    value: float
    anomaly_score: float
    is_anomaly: bool
    severity: str
    reason: str
    model: str
    inference_ms: float


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reading_id: int
    ts: str
    sensor_id: str
    severity: str
    score: float
    reason: Optional[str]
    acknowledged: bool
    status: str
    operator_note: Optional[str] = None
    resolved_at: Optional[str] = None
    feedback: Optional[str] = None


class MetricsResponse(BaseModel):
    total_readings: int
    total_anomalies: int
    anomaly_rate: float
    avg_inference_latency_ms: float
    current_model: str
    stream_rate_rpm: int


class AlertStatusUpdate(BaseModel):
    status: str
    operator_note: Optional[str] = None


class AlertFeedback(BaseModel):
    feedback: str  # e.g. "true_anomaly", "false_alarm", "unsure"


from typing import Literal

FAULT_TYPES = [
    "spike",
    "gradual_drift",
    "sensor_stuck",
    "missing_values",
    "noise_burst",
    "overheating",
    "vibration_fault"
]

class FaultInjectRequest(BaseModel):
    fault_type: str
    duration_steps: int = 10
    magnitude: float = 0.0
    sensor_id: str = "machine_temperature"

class FaultStatus(BaseModel):
    active: bool
    fault_type: Optional[str] = None
    sensor_id: Optional[str] = None
    step: int = 0
    duration_steps: int = 0
    magnitude: float = 0.0
