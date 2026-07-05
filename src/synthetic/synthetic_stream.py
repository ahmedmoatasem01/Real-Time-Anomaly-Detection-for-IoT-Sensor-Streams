from typing import Optional
from src.synthetic.fault_generator import FaultGenerator

class FaultManager:
    def __init__(self):
        self.active_generator: Optional[FaultGenerator] = None

    def start_fault(self, fault_type: str, duration_steps: int, magnitude: float, sensor_id: str) -> dict:
        self.active_generator = FaultGenerator(fault_type, duration_steps, magnitude, sensor_id)
        return self.active_generator.status()

    def stop_fault(self) -> dict:
        self.active_generator = None
        return {
            "active": False,
            "fault_type": None,
            "sensor_id": None,
            "step": 0,
            "duration_steps": 0,
            "magnitude": 0.0
        }

    def get_status(self) -> dict:
        if self.active_generator and not self.active_generator.finished:
            return self.active_generator.status()
        return self.stop_fault()

    def apply(self, current_value: float) -> float | None:
        """
        Called by the inference service / predict pipeline.
        Returns the modified value, or None if the sensor should emit missing_values.
        """
        if self.active_generator and not self.active_generator.finished:
            return self.active_generator.next_value(current_value)
        return current_value

# Singleton
fault_manager = FaultManager()
