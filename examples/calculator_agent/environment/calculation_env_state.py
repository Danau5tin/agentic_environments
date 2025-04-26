
from dataclasses import dataclass


@dataclass
class CalculatorState:
    """State for calculator environment."""
    consecutive_error_count: int = 0

    def copy(self) -> "CalculatorState":
        """Create a deep copy of the current state."""
        return CalculatorState(
            consecutive_error_count=self.consecutive_error_count
        )