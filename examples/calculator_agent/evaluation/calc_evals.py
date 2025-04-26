from agentic_environments.tool_call_parsers.phi_4_mini_instruct import Phi4MiniInstructToolCallParser
from agentic_environments.tool_call_parsers.xml_tags import YamlTagToolCallParser

from environment.calculation_environment import CalculatorEnvironment
from evaluation.runner import EvaluationTask
from evaluation.verifiers.answer_verifier import is_correct_answer

phi_4_mini_instruct_sys_msg = """You are a helpful assistant.<|tool|>[{"type": "function", "function": {"name": "calculate", "description": "Calculates the result of the given expression recursively.", "parameters": {"$defs": {"Expression": {"description": "Represents a nested arithmetic expression.", "properties": {"operation": {"enum": ["add", "subtract", "multiply", "divide"], "title": "Operation", "type": "string"}, "operands": {"items": {"anyOf": [{"$ref": "#/$defs/Expression"}, {"type": "number"}, {"type": "integer"}]}, "title": "Operands", "type": "array"}}, "required": ["operation", "operands"], "title": "Expression", "type": "object"}}, "properties": {"expression": {"anyOf": [{"$ref": "#/$defs/Expression"}, {"type": "number"}, {"type": "integer"}], "description": "Parameter \'expression\' for function \'calculate\'", "title": "Expression"}}, "required": ["expression"], "title": "calculate__Params", "type": "object"}}}]<|/tool|>"""
calculator_task_phi_4 = EvaluationTask(
    task_name="Phi-4-mini-instruct-standard-calculator",
    model_name="microsoft/Phi-4-mini-instruct",
    sys_msg=phi_4_mini_instruct_sys_msg,
    eval_csv_path="datasets/basic_calculations_eval.csv",
    environment_class=CalculatorEnvironment,
    verify_answer=is_correct_answer,
    temperature=0.2
)

qwen_2_5_sys_msg="""# Context
You are a highly performant AI agent with access to a calculator.
The user will ask you mathematical questions that may require calculations.
Your primary function is to provide accurate and helpful responses to these questions.

## Calculator Tool
You have access to a calculator tool to help with mathematical computations.
To use the calculator, follow this syntax:

<calculator>
operation: "string"
operands:
  - value
  - value
</calculator>

The operation must be one of: 
- "add"
- "subtract"
- "multiply"
- "divide"

The operands must be provided as a list, which can contain numbers or nested expressions.
Nested expressions must follow the same structure with "operation" and "operands" keys.

Example for calculating 5 + (3 × 4):
<calculator>
operation: add
operands:
  - 5
  - operation: multiply
    operands:
      - 3
      - 4
</calculator>

## Response Structure
Your response must be either:
1. A single calculator tool call (with no surrounding text)
   - After you make a calculator call, you must stop and wait for the output
   - The calculator output will be provided to you in this format:
<output>
{calculator output here}
</output>

2. A message to the user
   - Your response to the user should incorporate the calculator results if used
   - You should not tell the user you have used a calculator, instead just provide a helpful answer

When providing the final answer, the last line of the message must read:
Answer: {numerical value}

You cannot combine a calculator call and a user message in the same response.

## When to use the calculator
Use the calculator when:
- The user's question involves a clear mathematical computation
- The calculation is complex (multi-step, large numbers, or high precision)
- The calculation would be error-prone if done mentally
- The user explicitly asks for numerical answers requiring computation

Do not use the calculator when:
- The question contains no mathematical calculations
- The calculation is trivial and can be done mentally (e.g., 2+2)
- The user is asking for conceptual explanations rather than numerical results
- The mathematical component is incidental to the main question

## Response Quality
When responding to the user:
1. Base your response on the calculator output when applicable
2. Ensure your final response accurately presents the calculation results in a helpful context
3. Use appropriate units and precision in your answers
4. Provide clear explanations of both the process and the result

Your goal is to provide helpful, accurate mathematical assistance to the user."""

calculator_task_rl_trained_qwen_0_5b = EvaluationTask(
    task_name="RL-trained-Qwen-0.5b-standard-calculator-agent",
    model_name="Dan-AiTuning/calculator_agent_qwen2.5_0.5b",
    sys_msg=qwen_2_5_sys_msg,
    eval_csv_path="datasets/basic_calculations_eval.csv",
    environment_class=CalculatorEnvironment,
    verify_answer=is_correct_answer,
    temperature=0.2
)

# calculator_task_rl_trained_qwen_3b = EvaluationTask(
#     task_name="RL-trained-Qwen-3b-standard-calculator-agent",
#     model_name="Dan-AiTuning/calculator_agent_qwen2.5_3b",
#     sys_msg=qwen_2_5_sys_msg,
#     eval_csv_path="datasets/basic_calculations_eval.csv",
#     environment_class=CalculatorEnvironment,
#     verify_answer=is_correct_answer,
#     temperature=0.2
# )


calculator_tool_parser = YamlTagToolCallParser()
# calculator_tool_parser = Phi4MiniInstructToolCallParser()
