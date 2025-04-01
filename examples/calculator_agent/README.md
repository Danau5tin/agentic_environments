# Calculator Agent Example
This example project shows how `agentic_environments` can be extended for a particular use case.

Here I have extended `Environment`, built a tool, utilised `AgenticSystem` for inference, and built a `verifier` to confirm the final output of the model is correct.

## Use case
An agent which uses a calculator to help with arithmetic questions such as: "What is 567 * 2552 then divided by 887 then add 2828?"

## Environment, tools & state
The `CalculationEnvironment` simply ensures the tool `calculate` is executed, and the state `CalculatorState` is maintained.

## Evals

### Datasets
I utilised Gemini-2.5-Pro to help me build up my evaluation dataset of arithmetic problems and expressions. E.g:
```csv
prompt,expression
"what is 9*7?",9x7
```
Then I ran the csv through a script to generate the answer column. This script simply used `eval(expression)`.
This provided me with the prompt, answer format required to evaluate the model.

### Eval results
**Phi-4-Mini-Instruct Tool Calling Evaluation**

**Pass Rate:** 20.25% (32 out of 158 evaluations passed)

#### What I wanted to evaluate
- If the agent could handle a recursive argument schema in the tool call.
- If the agent would use the calculator tool all the time when it knows it has access to it.

#### Key Failure Insights:

- Tool Call Syntax/Schema Errors:
    - The most common failure mode involved generating invalid JSON or tool calls that didn't match the required schema (e.g., incorrect nesting, invalid operand formats, improper handling of multiple calls).
- Incorrect Logic/Order of Operations:
    - The model frequently misinterpreted the order of operations or the structure of multi-step calculations when translating natural language prompts into nested tool calls.
- Tool Avoidance:
    - In some cases, the model attempted to explain the calculation or perform it manually instead of using the provided tool, leading to failures.