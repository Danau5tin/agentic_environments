
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from agentic_environments.environment import Environment
from agentic_environments.model_output import ModelOutput


@dataclass
class Conversation:
    msgs: List[dict] = field(default_factory=list)

    def __post_init__(self):
        for msg in self.msgs:
            if not self.is_valid_msg_dict(msg):
                raise ValueError("Need role and content keys")
        return None
                
    def add_msg(self, msg: Dict[str, str]):
        if msg is None:
            return
        if not self.is_valid_msg_dict(msg):
            raise ValueError("Need role and content keys")
        self.msgs.append(msg)

    @staticmethod
    def is_valid_msg_dict(msg: dict) -> bool:
        return "role" in msg and "content" in msg


@dataclass
class FinishedConversation:
    conversation: Conversation
    environment_state: Any


class AgenticSystem:
    """
    Generic agent execution loop that manages the interaction between
    an agent and its environment.
    """
    
    def __init__(
        self,
        environment: Environment,
        agent_callback: Callable[[Conversation], ModelOutput],
        max_iterations: int = 10
    ):
        """
        Initialize the agent loop.
        
        Args:
            environment: Environment to execute actions in
            agent_callback: Function that produces agent outputs from state
            max_iterations: Maximum number of loop iterations
        """
        self.environment = environment
        self.agent_callback = agent_callback
        self.max_iterations = max_iterations
        self.current_iteration = 0
    
    def run(self, conversation: Conversation) -> FinishedConversation:
        """
        Run the agent loop until completion or max iterations.
        
        Args:
            initial_state: Starting state
            
        Returns:
            Final state after execution
        """
        self.current_iteration += 1
        
        model_output = self.agent_callback(conversation)
        conversation.add_msg(model_output.to_msg_dict())
        
        env_result = self.environment.handle_output(model_output)
        conversation.add_msg(env_result.resp_msg)
    
        if env_result.should_end_sequence or self.current_iteration >= self.max_iterations:
            finished_convo = FinishedConversation(
                conversation=conversation,
                environment_state=self.environment.get_state()
            )
            self.environment.cleanup()
            return finished_convo
            
        return self.run(conversation)