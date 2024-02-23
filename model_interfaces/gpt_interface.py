import json

from model_interfaces.interface import ChatSession
from utils.constants import ResetPolicy
from utils.openai import (
    ask_llm,
    LLMContext,
    make_system_message,
    make_user_message,
    make_assistant_message,
    get_max_prompt_size,
)
from dataclasses import dataclass, field


@dataclass
class GPTChatSession(ChatSession):
    system_prompt: str = "You are a helpful assistant."
    max_prompt_size: int = None
    model: str = "gpt-4"
    verbose: bool = False
    context: LLMContext = field(default_factory=LLMContext)

    def __post_init__(self):
        super().__post_init__()
        if len(self.context) == 0:
            self.context.append(make_system_message(self.system_prompt))
        if self.max_prompt_size is None:
            self.max_prompt_size = get_max_prompt_size(self.model)
        else:
            self.max_prompt_size = min(self.max_prompt_size, get_max_prompt_size(self.model))

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size}"

    def reply(self, user_message: str) -> str:
        self.context.append(make_user_message(user_message))
        if self.verbose:
            print(f"USER: {user_message}")

        def cost_callback(cost_usd: float):
            self.costs_usd += cost_usd

        response = ask_llm(
            self.context,
            model=self.model,
            temperature=0,
            context_length=self.max_prompt_size,
            cost_callback=cost_callback,
        )
        if self.verbose:
            print(f"AGENT: {response}")
        self.context.append(make_assistant_message(response))
        return response

    def reset(self):
        self.context = [make_system_message(self.system_prompt)]

    def save(self):
        fname = self.save_path.joinpath("context.json")
        with open(fname, "w") as fd:
            json.dump(self.context, fd)

    def load(self):
        fname = self.save_path.joinpath("context.json")
        with open(fname, "r") as fd:
            self.context = json.load(fd)
