from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import litellm

from ltm.agent import LTMAgent
# from ltm.agent_think import LTMReflexionAgent
from utils.llm import count_tokens_for_model
litellm.modify_params = True  # To allow it adjusting the prompt for Claude LLMs
from model_interfaces.interface import ChatSession


@dataclass
class LTMAgentWrapper(ChatSession):
    model: str = None
    max_prompt_size: int = None
    max_message_size: int = 1024

    def __post_init__(self):
        self.agent = LTMAgent(
            run_name=self.run_name, model=self.model,
            max_prompt_size=self.max_prompt_size, max_completion_tokens=self.max_message_size,
        )
        self.retrieval_evaluator = self.agent.retrieval_evaluator

    @property
    def name(self):
        model = self.model.replace("/", "-")
        return f"{super().name} - {model} - {self.max_prompt_size} - {self.max_message_size}"

    @property
    def state_path(self) -> Path:
        return self.save_path.joinpath("full_agent.json")

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        def _cost_fn(amount: float):
            self.costs_usd += amount

        return self.agent.reply(user_message, agent_response=agent_response, cost_callback=_cost_fn)

    def reset(self):
        self.agent.reset()

    def save(self):
        with open(self.state_path, "w") as fd:
            fd.write(self.agent.state_as_text())

    def load(self):
        with open(self.state_path, "r") as fd:
            self.agent.from_state_text(fd.read())

    def token_len(self, text: str) -> int:
        return count_tokens_for_model(model=self.model, text=text)