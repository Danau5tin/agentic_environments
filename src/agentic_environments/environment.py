from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Optional

from agentic_environments.model_output import ModelOutput
from agentic_environments.state import STATE



@dataclass
class EnvironmentResult:
    """Result of handling an action in the environment."""

    should_end_sequence: bool = False
    resp_msg: Optional[Dict[str, str]] = None
    exception: Optional[Exception] = None

    @property
    def has_error(self) -> bool:
        return self.exception is not None


class Environment(Generic[STATE], ABC):
    """Abstract environment interface that can handle agent outputs and maintain its own state if required.

    It will receive agent outputs and is responsible for updating the environment state and returning a response for the agent if required.
    """

    def __init__(self, env_idx: int = 0):
        self.env_idx = env_idx

    @abstractmethod
    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        """
        This should handle:
        1. Tool call execution
        2. Internal state updates (if any)
        3. Deciding when the environment (& therefore the sequence generation) should end

        Args:
            model_output: Output from the agent

        Returns:
            EnvironmentResult
        """

    @abstractmethod
    def get_state(self) -> Optional[STATE]:
        """Get the current state of this environment."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources when done with this environment."""
