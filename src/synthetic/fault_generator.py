import random

FAULT_TYPES = [
    "spike",
    "gradual_drift",
    "sensor_stuck",
    "missing_values",
    "noise_burst",
    "overheating",
    "vibration_fault"
]

class FaultGenerator:
    def __init__(self, fault_type: str, duration_steps: int, magnitude: float, sensor_id: str, baseline: float = 0.0):
        self.fault_type = fault_type
        self.duration_steps = duration_steps
        self.magnitude = magnitude
        self.sensor_id = sensor_id
        self.baseline = baseline
        self.step = 0

    def next_value(self, current_real_value: float) -> float | None:
        if self.finished:
            return current_real_value
            
        if self.fault_type == "spike":
            # First few steps are spiked
            val = current_real_value + self.magnitude if self.step < min(3, self.duration_steps) else current_real_value
        elif self.fault_type == "gradual_drift":
            ratio = self.step / max(1, self.duration_steps)
            val = current_real_value + (self.magnitude * ratio)
        elif self.fault_type == "sensor_stuck":
            if self.step == 0:
                self.baseline = current_real_value
            val = self.baseline
        elif self.fault_type == "missing_values":
            val = None
        elif self.fault_type == "noise_burst":
            # Gauss around 0 with std dev = magnitude
            val = current_real_value + random.gauss(0, self.magnitude)
        elif self.fault_type == "overheating":
            ratio = self.step / max(1, self.duration_steps)
            val = current_real_value + (self.magnitude * (ratio ** 2))
        elif self.fault_type == "vibration_fault":
            val = current_real_value
        else:
            val = current_real_value

        self.step += 1
        return val

    @property
    def finished(self) -> bool:
        return self.step >= self.duration_steps

    def status(self) -> dict:
        return {
            "fault_type": self.fault_type,
            "sensor_id": self.sensor_id,
            "step": self.step,
            "duration_steps": self.duration_steps,
            "magnitude": self.magnitude,
            "active": not self.finished
        }
