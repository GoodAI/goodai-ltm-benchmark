from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import litellm
from utils.llm import count_tokens_for_model
litellm.modify_params = True  # To allow it adjusting the prompt for Claude LLMs
from ltm.agent import LTMAgent
from model_interfaces.interface import ChatSession


@dataclass
class LTMAgentWrapper(ChatSession):
    model: str = None
    max_prompt_size: int = None

    def __post_init__(self):
        self.agent = LTMAgent(
            model=self.model, max_prompt_size=self.max_prompt_size, max_completion_tokens=self.max_message_size,
        )

    @property
    def name(self):
        model = self.model.replace("/", "-")
        return f"{super().name} - {model} - {self.max_prompt_size}"

    @property
    def state_path(self) -> Path:
        return self.save_path.joinpath("full_agent.json")

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        def _cost_fn(amount: float):
            self.costs_usd += amount

        return self.agent.reply(user_message, cost_callback=None if "llama" in self.model else _cost_fn)

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