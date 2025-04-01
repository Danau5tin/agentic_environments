
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CalculatorState:
    """State for calculator environment."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    consecutive_error_count: int = 0

    def copy(self) -> "CalculatorState":
        """Create a deep copy of the current state."""
        return CalculatorState(
            messages=self.messages.copy(),
            consecutive_error_count=self.consecutive_error_count
        )

    def add_message(self, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add a message to the state."""
        message = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)
