# Agentic Environments

A lightweight, extensible framework for building tool-calling LLM agents.

[![PyPI version](https://badge.fury.io/py/agentic-environments.svg)](https://badge.fury.io/py/agentic-environments)

## Overview

Agentic Environments provides a simple yet powerful framework for creating and managing the interaction between language models and their environments. It makes it easy to build agentic systems where LLMs can use tools, manage state, and interact with external systems through a consistent interface.

## Features

- 🚀 **Extensible Environments**: Define custom environments that your agents can interact with
- 🔄 **Flexible Agent Loop**: Simple execution loop for agent-environment interaction
- 🧰 **Tool Call Parsing**: Built-in parsers for different LLM tool call formats
- 📦 **State Management**: Track and maintain environment state throughout agent interactions
- 🔌 **Model Agnostic**: Works with any LLM that can produce tool calls

## Installation

```bash
pip install agentic-environments
```

## Quick Start

Here's a simple example of how to use Agentic Environments:

```python
from agentic_environments.environment import Environment, EnvironmentResult
from agentic_environments.model_output import ModelOutput, ToolCall
from agentic_environments.conversation import Conversation
from agentic_environments.agentic_system import AgenticSystem

# 1. Create your custom environment
class CounterEnv(Environment):
    def __init__(self):
        super().__init__()
        self.count = 0
        
    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        if not model_output.tool_calls:
            return EnvironmentResult(should_end_sequence=True)
            
        for tool_call in model_output.tool_calls:
            if tool_call.tool_name == "increment_counter":
                try:
                    amount = tool_call.tool_parameters.get("amount", 1)
                    self.count += amount
                    return EnvironmentResult(
                        should_end_sequence=False,
                        resp_msg={"role": "user", "content": f"Counter is now: {self.count}"}
                    )
                except Exception as e:
                    return EnvironmentResult(
                        should_end_sequence=False,
                        resp_msg={"role": "user", "content": f"Try again. Error: {str(e)}"},
                        exception=e
                    )
        
        return EnvironmentResult(
            should_end_sequence=False,
            resp_msg={"role": "user", "content": "Unknown tool call."}
        )
    
    def get_state(self):
        return {"count": self.count}
    
    def cleanup(self):
        # No resources to clean up for this simple environment
        pass

# 2. Implement your agent callback
def agent_callback(conversation: Conversation) -> ModelOutput:
    # This is where you would call your LLM, for demonstration purposes, let's create a mock output
    
    # You can use one of the provided parsers or implement your own
    from agentic_environments.tool_call_parsers.xml_tags_with_yaml_content import XMLTagWithYamlContentToolCallParser
    parser = XMLTagWithYamlContentToolCallParser()
    
    # LLM response that contains a tool call
    llm_response = """    
    <increment_counter>
    amount: 5
    </increment_counter>
    """
    
    # Parse the tool calls from the response
    tool_calls = parser.parse_tool_calls(llm_response)
    
    # Create a ModelOutput
    return ModelOutput(raw_content=llm_response, tool_calls=tool_calls)

# 3. Initialize and run your agentic system
counter_env = CounterEnv()
system = AgenticSystem(
    environment=counter_env,
    agent_callback=agent_callback,
    max_iterations=5
)

# Create initial conversation
initial_conversation = Conversation(msgs=[
    {"role": "user", "content": "Please increment the counter by 5."}
])

# Run the agent
finished_conversation = system.run(initial_conversation)

# Access the final state and conversation
print("Final conversation:", finished_conversation.conversation.msgs)
print("Final state:", finished_conversation.environment_state)
```

## Core Components

### Environment

The `Environment` class is the core of the framework. It defines how your agent interacts with external systems and tools.

```python
from agentic_environments.environment import Environment, EnvironmentResult
from agentic_environments.model_output import ModelOutput
from typing import Optional, Any

class MyCustomEnvironment(Environment[dict]):
    def __init__(self):
        super().__init__()
        self.state = {}
        
    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        # Process tool calls and update state
        # Return appropriate response
        
    def get_state(self) -> Optional[dict]:
        return self.state
        
    def cleanup(self) -> None:
        # Clean up any resources
```

### AgenticSystem

The `AgenticSystem` manages the execution loop between your agent and its environment.

```python
from agentic_environments.agentic_system import AgenticSystem
from agentic_environments.conversation import Conversation

system = AgenticSystem(
    environment=my_environment,
    agent_callback=my_agent_function,
    max_iterations=10
)

result = system.run(initial_conversation)
```

### Tool Call Parsers

The framework includes parsers for different tool call formats:

1. **XMLTagWithYamlContentToolCallParser**: Parses tool calls in YAML format within custom XML tags
2. **StandardFunctionCallingToolCallParser**: Specialized parser for standard function calling models

You can implement your own parser by extending the `ToolCallParser` class:

```python
from agentic_environments.tool_call_parsers.tool_call_parser import ToolCallParser
from agentic_environments.model_output import ToolCall
from typing import List

class MyCustomParser(ToolCallParser):
    def parse_tool_calls(self, response_text: str) -> List[ToolCall]:
        # Parse tool calls from response_text
        # Return a list of ToolCall objects
```

## Building Your Own Agent

To create your own agent:

1. **Define your environment**: Extend the `Environment` class to create your custom environment
2. **Implement your agent callback**: Create a function that takes a `Conversation` and returns a `ModelOutput`
3. **Choose or create a parser**: Select one of the built-in parsers or implement your own
4. **Set up the agentic system**: Initialize and run the system with your components

## Advanced Usage

### Docker Environment Example

```python
class DockerEnv(Environment[FileSystemState]):
    def __init__(self, port_map: dict):
        super().__init__()
        self.port_map = port_map
        self.container = None
        self.file_system = FileSystem()
        
    def handle_output(self, model_output: ModelOutput) -> EnvironmentResult:
        # Process Docker-related tool calls
        # Update file system state
        # Return appropriate response
        
    def get_state(self) -> FileSystemState:
        return self.file_system.to_dict()
        
    def cleanup(self) -> None:
        # Stop and remove Docker container
        if self.container:
            self.container.stop()
            self.container.remove()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.