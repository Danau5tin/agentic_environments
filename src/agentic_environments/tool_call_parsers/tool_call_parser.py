from abc import ABC
from typing import List

from agentic_environments.model_output import ToolCall



class ToolCallParser(ABC):
    
    def parse_tool_calls(self, response_text: str) -> List[ToolCall]:
        raise NotImplementedError()
