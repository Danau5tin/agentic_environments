# Agentic Enviroments
A mini-framework for tool calling LLM agents.

## Features
- Easily extensible `Environment`s
- Simple agentic inference & environment interaction with `AgenticSystem`

## Example usage
```python
class DockerEnv[FileSystemState](Environment):

    def __init__(self, port_map: dict):
        # Docker setup

    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        # handle tool calls with model_output.tool_calls
        # update internal env state
        return EnvironmentResult(
            should_end_sequence=False,
            output_to_show_model="output",
        )
    
    def get_state(self) -> FileSystemState:
        return self.file_system.to_dict()
    
    def clean_up(self) -> None:
        # Clean up resources


def agent_callback(state: FileSystemState) -> ModelOutput:
    # Run inference using transformers, vLLM, etc..
    # Parse tool calls to ModelOutput depending on the model/service used
    return ModelOutput(
        raw_output="tool(k=v)", 
        tool_calls=[ToolCall(name="tool", parameters={"k":"v"})]
    )

docker_env = DockerEnv(port_map={"8000:8000"})

system = AgenticSystem[FileSystemState](
    environment=docker_env,
    agent_callback=agent_callback,
    max_iterations=10,
)

initial_state = docker_env.get_state()
system.run()
```

## Actual example
In `examples/calculator_agent` I have challenged Phi-4-mini-instruct with trying to use a calculator tool with a not so straightforward tool call API.

This uses an extended `Environment`, custom tools, `AgenticSystem`, as well as transformers library for inference.