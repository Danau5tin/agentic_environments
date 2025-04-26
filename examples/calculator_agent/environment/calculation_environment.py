import logging
from typing import List

from agentic_environments.environment import Environment, EnvironmentResult
from agentic_environments.model_output import ModelOutput, ToolCall
from environment.calculation_env_state import CalculatorState

from dataclasses import dataclass
from typing import Any, List, Literal, Union, Dict, Callable


@dataclass
class Expression:
    """Represents a mathematical expression or sub-expression."""

    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
    ]
    operands: List[Union["Expression", float, int]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Expression':
        """Parse a dictionary to create an Expression object."""
        
        # Extract the operation and ensure it's valid
        operation = data.get("operation")
        if operation not in {
            "add", "subtract", "multiply", "divide",
        }:
            raise ValueError(f"Invalid operation: {operation}")

        # Extract operands, which can be nested expressions or numbers
        operands_data = data.get("operands")
        if not isinstance(operands_data, list):
            raise ValueError("Operands must be a list.")

        operands = []
        for operand in operands_data:
            if isinstance(operand, (int, float)):
                operands.append(operand)
            elif isinstance(operand, dict):
                # Recursively parse nested expressions
                operands.append(cls.from_dict(operand))
            else:
                raise ValueError("Operands must be numbers or dictionaries representing expressions.")

        return cls(operation=operation, operands=operands)




def _add(operands: List[float]) -> float:
    """Adds all operands. Returns 0 for empty list."""
    return sum(operands)

def _subtract(operands: List[float]) -> float:
    """Subtracts subsequent operands from the first.
       Returns 0 for empty list. Returns the operand if only one."""
    if not operands:
        return 0.0
    result = operands[0]
    for operand in operands[1:]:
        result -= operand
    return result

def _multiply(operands: List[float]) -> float:
    """Multiplies all operands. Returns 1 for empty list (identity)."""
    result = 1.0
    for operand in operands:
        result *= operand
    return result

def _divide(operands: List[float]) -> float:
    """Divides the first operand by subsequent operands.
       Returns the operand if only one. Raises error for empty list."""
    if not operands:
        raise ValueError("Division requires at least one operand")
    if len(operands) == 1:
        return operands[0]

    result = operands[0]
    for operand in operands[1:]:
        if operand == 0.0:
            raise ZeroDivisionError("Division by zero")
        result /= operand
    return result

_operations: Dict[str, Callable[[List[float]], float]] = {
    "add": _add,
    "subtract": _subtract,
    "multiply": _multiply,
    "divide": _divide,
}

def calculate(expression: Union[Expression, float, int]) -> float:
    """
    Calculates the result of the given expression recursively.

    Args:
        expression: An Expression object, float, or int to evaluate.

    Returns:
        The calculated result as a float.

    Raises:
        ValueError: If an unsupported operation is encountered or if
                    an operation receives an invalid number of operands.
        ZeroDivisionError: If division by zero occurs.
        TypeError: If the input is not an Expression, float, or int.
    """
    if isinstance(expression, (int, float)):
        return float(expression)

    if isinstance(expression, Expression):
        evaluated_operands: List[float] = [calculate(operand) for operand in expression.operands]

        operation_func = _operations.get(expression.operation)
        if operation_func is None:
            raise ValueError(f"Unsupported operation: {expression.operation}")

        return operation_func(evaluated_operands)

    raise TypeError(f"Unsupported expression type: {type(expression)}")



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
    SUPPORTED_TOOL = "calculator"

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
            expression = Expression.from_dict(tool_call.tool_parameters)
            result = calculate(expression)

            return EnvironmentResult(
                should_end_sequence=False,
                resp_msg={
                    "role": "user",
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
                "role": "user",
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
        