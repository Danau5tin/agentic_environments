from agentic_environments.tool_call_parsers.phi_4_mini_instruct import Phi4MiniInstructToolCallParser

from environment.calculation_env_state import CalculatorState
from environment.calculation_environment import CalculatorEnvironment
from evaluation.runner import EvaluationTask
from evaluation.verifiers.answer_verifier import is_correct_answer


def create_calculator_initial_state(prompt: str) -> CalculatorState:
    """Creates the initial state for the calculator task."""

    phi_4_mini_instruct_sys_msg = """You are a helpful assistant.<|tool|>[{"type": "function", "function": {"name": "calculate", "description": "Calculates the result of the given expression recursively.", "parameters": {"$defs": {"Expression": {"description": "Represents a nested arithmetic expression.", "properties": {"operation": {"enum": ["add", "subtract", "multiply", "divide"], "title": "Operation", "type": "string"}, "operands": {"items": {"anyOf": [{"$ref": "#/$defs/Expression"}, {"type": "number"}, {"type": "integer"}]}, "title": "Operands", "type": "array"}}, "required": ["operation", "operands"], "title": "Expression", "type": "object"}}, "properties": {"expression": {"anyOf": [{"$ref": "#/$defs/Expression"}, {"type": "number"}, {"type": "integer"}], "description": "Parameter \'expression\' for function \'calculate\'", "title": "Expression"}}, "required": ["expression"], "title": "calculate__Params", "type": "object"}}}]<|/tool|>"""
    state = CalculatorState(
        messages=[
            {"role": "system", "content": phi_4_mini_instruct_sys_msg},
            {"role": "user", "content": prompt},
        ]
    )
    return state


calculator_task = EvaluationTask(
    task_name="Phi-4-mini-instruct-standard-calculator",
    model_name="microsoft/Phi-4-mini-instruct",
    eval_csv_path="datasets/basic_calculations_eval.csv",
    environment_class=CalculatorEnvironment,
    create_initial_state=create_calculator_initial_state,
    verify_answer=is_correct_answer,
    temperature=0.2
)


calculator_tool_parser = Phi4MiniInstructToolCallParser()
