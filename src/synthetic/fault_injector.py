from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("fault_injector")

class FaultInjector:
    def __init__(self):
        self.active_fault: Optional[str] = None
        self.magnitude: float = 0.0
        self.steps_remaining: int = 0
        self.base_value: float = 0.0

    def inject(self, fault_type: str, magnitude: float, duration_steps: int, current_value: float):
        """
        Activates a new fault.
        """
        self.active_fault = fault_type
        self.magnitude = magnitude
        self.steps_remaining = duration_steps
        self.base_value = current_value
        logger.info(f"Injecting fault '{fault_type}', mag={magnitude}, steps={duration_steps}")

    def apply(self, value: float) -> float:
        """
        Applies the active fault to the current reading if a fault is active.
        """
        if self.steps_remaining <= 0:
            self.active_fault = None
            return value
            
        modified_value = value
        
        if self.active_fault == "spike":
            # Add magnitude directly to the value
            modified_value = value + self.magnitude
        elif self.active_fault == "drift":
            # Progressive drift: base + (magnitude * step_ratio)
            # wait, if we are at step 1 of 10, ratio = 1/10
            # Since steps_remaining decreases from duration_steps to 0, we'd need to know the initial duration.
            # We'll just add magnitude to the value cumulatively, but wait, apply() doesn't track state easily
            # without knowing the initial steps. Let's just add magnitude * (1 / steps_remaining) or something.
            # Better: a constant drift added each step.
            modified_value = value + self.magnitude
            self.magnitude *= 1.1  # compound it
        elif self.active_fault == "stuck":
            # Returns the base value constantly
            modified_value = self.base_value
        elif self.active_fault == "dropout":
            # Drop to zero
            modified_value = 0.0
        elif self.active_fault == "overheating":
            # Similar to drift, but rapid rise
            modified_value = value + self.magnitude
            self.magnitude += 2.0  # linear increase
            
        self.steps_remaining -= 1
        if self.steps_remaining <= 0:
            logger.info(f"Fault '{self.active_fault}' concluded.")
            self.active_fault = None
            
        return float(modified_value)
