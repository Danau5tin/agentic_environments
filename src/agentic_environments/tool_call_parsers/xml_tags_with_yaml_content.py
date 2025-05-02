import re
import yaml
from typing import List, Any

from agentic_environments.model_output import ToolCall
from agentic_environments.tool_call_parsers.tool_call_parser import ToolCallParser

class XMLTagWithYamlContentToolCallParser(ToolCallParser):
    """
    Parser for extracting tool calls from model outputs using YAML within custom tags.

    Expects formats like:
    <tool_name>
    yaml_content
    </tool_name>

    Where 'yaml_content' is a valid YAML string representing the tool parameters.
    """

    def parse_tool_calls(self, response_text: str) -> List[ToolCall]:
        """
        Parse tool calls from the response text based on <tag>yaml</tag> format.

        Args:
            response_text (str): The raw text output from the model.

        Returns:
            List[ToolCall]: List of parsed tool calls, empty if none found or
                            if YAML parsing fails.
        """
        tool_calls: List[ToolCall] = []

        # Regex explanation:
        # <(\w+?)>   : Match opening tag like <tool_name>, capture 'tool_name' in group 1 (non-greedy)
        # (.*?)      : Match any character (including newlines) between tags, capture in group 2 (non-greedy)
        # </\1>      : Match the corresponding closing tag using a backreference to group 1
        # re.DOTALL  : Make '.' match newline characters
        pattern = r'<(\w+?)>(.*?)</\1>'
        matches = re.finditer(pattern, response_text, re.DOTALL)

        for match in matches:
            tool_name = match.group(1)
            yaml_content = match.group(2).strip() 

            if not tool_name or not yaml_content:
                continue

            try:
                arguments: Any = yaml.safe_load(yaml_content)

                if isinstance(arguments, dict):
                    tool_call = ToolCall(
                        tool_name=tool_name,
                        tool_parameters=arguments
                    )
                    tool_calls.append(tool_call)
                else:
                    print(f"Warning: Tool '{tool_name}' has non-dict parameters. Skipping.")
                    continue
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse YAML for tool '{tool_name}'. Error: {e}")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error processing tool '{tool_name}'. Error: {e}")
                continue

        return tool_calls
    