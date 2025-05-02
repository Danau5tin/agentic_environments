import re
import json
from typing import List

from agentic_environments.model_output import ToolCall
from agentic_environments.tool_call_parsers.tool_call_parser import ToolCallParser

class StandardFunctionCallingToolCallParser(ToolCallParser):
    """
    Parser for extracting tool calls from tool calling models that accept function calling in the chat template JSON format.
    Ollama maintains a great list [here](https://ollama.com/search?c=tools)
    Example models include:
     - Phi-4-mini
     - Llama 3.3
     - Qwen2.5 & 3
    
    Supports two output formats:
    1. <|tool_call|>[{"type": "function", "function": {"name": "tool_name", "arguments": {...}}}]<|/tool_call|>
    2. <|tool_call|>[{"name": "tool_name", "arguments": {...}}]<|/tool_call|>
    """
    
    def parse_tool_calls(self, response_text: str) -> List[ToolCall]:
        """
        Parse tool calls from the response text.
        
        Args:
            response_text (str): The raw text output from the model
            
        Returns:
            List[ToolCall]: List of parsed tool calls, empty if none found
        """
        tool_calls = []
        
        pattern = r'<\|tool_call\|>(.*?)<\|/tool_call\|>'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                tool_call_data = json.loads(match)
                
                if isinstance(tool_call_data, list):
                    tool_calls.extend(self._process_tool_call_list(tool_call_data))
                else:
                    tool_call = self._process_single_tool_call(tool_call_data)
                    if tool_call:
                        tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
            except Exception:
                # Skip any other parsing errors
                continue
                
        return tool_calls
    
    def _process_tool_call_list(self, tool_call_list: List[dict]) -> List[ToolCall]:
        result = []
        for tool_call_data in tool_call_list:
            tool_call = self._process_single_tool_call(tool_call_data)
            if tool_call:
                result.append(tool_call)
        return result
    
    def _process_single_tool_call(self, tool_call_data: dict) -> ToolCall:
        if not isinstance(tool_call_data, dict):
            return None
            
        # Format 1: {"type": "function", "function": {"name": "tool_name", "arguments": {...}}}
        if 'type' in tool_call_data and tool_call_data.get('type') == 'function' and 'function' in tool_call_data:
            function_data = tool_call_data['function']
            if 'name' in function_data and 'arguments' in function_data:
                return self._create_tool_call(function_data['name'], function_data['arguments'])
        
        # Format 2: {"name": "tool_name", "arguments": {...}}
        elif 'name' in tool_call_data and 'arguments' in tool_call_data:
            return self._create_tool_call(tool_call_data['name'], tool_call_data['arguments'])
        
        return None
    
    def _create_tool_call(self, tool_name: str, arguments) -> ToolCall:
        """Helper method to create a ToolCall with proper argument handling"""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        
        return ToolCall(
            tool_name=tool_name,
            tool_parameters=arguments
        )