import logging
from typing import List, Optional

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

    def __init__(self, initial_state: Optional[CalculatorState] = None, env_idx: int = 0):
        super().__init__(env_idx=env_idx)
        self.state = initial_state or CalculatorState()
        self.logger = logging.getLogger(f"{__name__}.CalculatorEnv[{env_idx}]")
        self.logger.setLevel(logging.INFO)

    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        """Process model output and update environment state."""
        new_state = self.state.copy()
        
        tool_calls_dict = [tc.to_dict() for tc in model_output.tool_calls] if model_output.tool_calls else None
        new_state.add_message("assistant", model_output.raw_content, tool_calls_dict)

        # If no tool calls, consider it a final answer
        if not model_output.tool_calls:
            self.logger.debug("Model provided final answer.")
            self.state = new_state
            return EnvironmentResult(should_end_sequence=True)

        result = self._handle_tool_calls(model_output.tool_calls, new_state)
        self._update_error_count(new_state, result.has_error)
        
        if new_state.consecutive_error_count >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.warning(f"Max consecutive errors ({self.MAX_CONSECUTIVE_ERRORS}) reached. Forcing end.")
            result.should_end_sequence = True

        self.state = new_state
        return result

    def _update_error_count(self, state: CalculatorState, has_error: bool) -> None:
        """Update the consecutive error count based on result."""
        if has_error:
            state.consecutive_error_count += 1
            self.logger.debug(f"Error occurred. Consecutive error count: {state.consecutive_error_count}")
        else:
            if state.consecutive_error_count > 0:
                self.logger.debug("Successful tool call. Resetting consecutive error count.")
            state.consecutive_error_count = 0

    def _handle_tool_calls(self, tool_calls: List[ToolCall], state: CalculatorState) -> EnvironmentResult:
        """Process tool calls and return result."""
        try:
            if len(tool_calls) > 1:
                raise MultipleToolCallsError("Multiple tool calls not supported. Try one at a time.")

            tool_call = tool_calls[0]
            if tool_call.tool_name != self.SUPPORTED_TOOL:
                raise UnsupportedToolCallError(f"Unsupported tool call: {tool_call.tool_name}")

            return self._execute_calculator_call(tool_call, state)

        except CalculatorError as e:
            return self._create_error_result(e, state)
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return self._create_error_result(e, state)

    def _execute_calculator_call(self, tool_call: ToolCall, state: CalculatorState) -> EnvironmentResult:
        """Execute calculator tool call and handle results."""
        try:
            expression = Expression(**tool_call.tool_parameters["expression"])
            result = calculate(expression)
            
            tool_call_output = str(result)
            state.add_message("tool", tool_call_output)
            
            return EnvironmentResult(
                should_end_sequence=False,
                output_to_show_model=tool_call_output,
            )
        except Exception as e:
            return self._create_error_result(e, state)

    def _create_error_result(self, exception: Exception, state: CalculatorState) -> EnvironmentResult:
        """Create and record an error result with appropriate truncation."""
        error_msg = f"Error: {str(exception)}"
        if len(error_msg) > self.MAX_ERROR_MSG_LENGTH:
            error_msg = f"Error: Output too long. Truncated: {str(exception)[:self.MAX_ERROR_MSG_LENGTH]}..."
        
        state.add_message("tool", error_msg)
        
        return EnvironmentResult(
            should_end_sequence=False,
            output_to_show_model=error_msg,
            exception=exception,
        )

    def get_state(self) -> CalculatorState:
        """Return the current state."""
        return self.state

    def cleanup(self):
        """Clean up resources when done."""
        # Nothing to clean up
        