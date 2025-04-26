from typing import Dict, List, Optional

class ToolCall:
    """
    A ToolCall represents a tool call that the agent made.
    """

    def __init__(self, tool_name: str, tool_parameters: dict[str, any]):
        self.tool_name = tool_name
        self.tool_parameters = tool_parameters

    def to_dict(self) -> dict:
        return {
            "name": self.tool_name,
            "parameters": self.tool_parameters,
        }

class ModelOutput:
    """
    A ModelOutput contains the raw content of the output and any tool calls made by the agent.

    It signifies a single output from the model after inference.
    """

    def __init__(self, raw_content: str, tool_calls: Optional[List[ToolCall]]):
        self.raw_content = raw_content
        self.tool_calls = tool_calls

    def to_msg_dict(self) -> Dict[str, str]:
        assistant_msg = {
            "role": "assistant",
            "content": self.raw_content,
        }
        if self.tool_calls:
            tool_call_dicts = [tc.to_dict() for tc in self.tool_calls]
            assistant_msg["tool_calls"] = tool_call_dicts

        return assistant_msg
