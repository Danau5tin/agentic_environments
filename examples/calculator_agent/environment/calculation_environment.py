import logging
from typing import List

from agentic_environments.environment import Environment, EnvironmentResult
from agentic_environments.model_output import ModelOutput, ToolCall
from environment.calculation_env_state import CalculatorState
from environment.tools.calculator import calculate, Expression


class CalculatorError(Exception):
    """Base exception for calculator environment errors."""


class UnsupportedToolCallError(CalculatorError):
    """Exception raised when an unsupported tool call is made."""


class MultipleToolCallsError(CalculatorError):
    """Exception raised when multiple tool calls are made simultaneously."""



class CalculatorEnvironment(Environment):
    """Environment for calculator agent."""

    MAX_CONSECUTIVE_ERRORS = 2
    MAX_ERROR_MSG_LENGTH = 250
    SUPPORTED_TOOL = "calculate"

    def __init__(self, env_idx: int = 0):
        super().__init__(env_idx=env_idx)
        self.state = CalculatorState()
        self.logger = logging.getLogger(f"{__name__}.CalculatorEnv[{env_idx}]")
        self.logger.setLevel(logging.INFO)

    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        """Process model output and update environment state."""
        
        if not model_output.tool_calls:
            self.logger.debug("Model provided final answer.")
            return EnvironmentResult(should_end_sequence=True)

        result = self._handle_tool_calls(model_output.tool_calls)
        self._update_error_count(result.has_error)
        
        if self.state.consecutive_error_count >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.warning(f"Max consecutive errors ({self.MAX_CONSECUTIVE_ERRORS}) reached. Forcing end.")
            result.should_end_sequence = True

        return result

    def _update_error_count(self, has_error: bool) -> None:
        """Update the consecutive error count based on result."""
        if has_error:
            self.state.consecutive_error_count += 1
            self.logger.debug(f"Error occurred. Consecutive error count: {self.state.consecutive_error_count}")
        else:
            if self.state.consecutive_error_count > 0:
                self.logger.debug("Successful tool call. Resetting consecutive error count.")
            self.state.consecutive_error_count = 0

    def _handle_tool_calls(self, tool_calls: List[ToolCall]) -> EnvironmentResult:
        """Process tool calls and return result."""
        try:
            if len(tool_calls) > 1:
                raise MultipleToolCallsError("Multiple tool calls not supported. Try one at a time.")

            tool_call = tool_calls[0]
            if tool_call.tool_name != self.SUPPORTED_TOOL:
                raise UnsupportedToolCallError(f"Unsupported tool call: {tool_call.tool_name}")

            return self._execute_calculator_call(tool_call)

        except CalculatorError as e:
            return self._create_error_result(e)
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return self._create_error_result(e)

    def _execute_calculator_call(self, tool_call: ToolCall) -> EnvironmentResult:
        """Execute calculator tool call and handle results."""
        try:
            expression = Expression(**tool_call.tool_parameters["expression"])
            result = calculate(expression)

            return EnvironmentResult(
                should_end_sequence=False,
                resp_msg={
                    "role": "tool",
                    "content": str(result),
                },
            )
        except Exception as e:
            return self._create_error_result(e)

    def _create_error_result(self, exception: Exception) -> EnvironmentResult:
        """Create and record an error result with appropriate truncation."""
        error_msg = f"Error: {str(exception)}"
        if len(error_msg) > self.MAX_ERROR_MSG_LENGTH:
            error_msg = f"Error: Output too long. Truncated: {str(exception)[:self.MAX_ERROR_MSG_LENGTH]}..."
        
        return EnvironmentResult(
            should_end_sequence=False,
            resp_msg={
                "role": "tool",
                "content": error_msg,
            },
            exception=exception,
        )

    def get_state(self) -> CalculatorState:
        """Return the current state."""
        return self.state

    def cleanup(self):
        """Clean up resources when done."""
        # Nothing to clean up
        