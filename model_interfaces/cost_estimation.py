from typing import Optional

from model_interfaces.interface import ChatSession
from dataclasses import dataclass, field
from utils.llm import (
    LLMContext,
    make_system_message,
    make_user_message,
    make_assistant_message,
    context_token_len, tokens_in_text,
)


@dataclass
class CostEstimationChatSession(ChatSession):
    max_prompt_size: int = 0
    cost_in_token: float = 0
    cost_out_token: float = 0
    avg_response_len: float = 106  # Computed from one of our runs (in tokens)
    context: LLMContext = field(default_factory=LLMContext)
    context_tokens: int = 0
    expected_response_tokens: int = 512

    @property
    def name(self) -> str:
        return f"{super().name} - {self.max_prompt_size} - {self.cost_in_token:.2e} - {self.cost_out_token:.2e}"

    def __post_init__(self):
        super().__post_init__()
        self.system_prompt = "You are a helpful assistant."
        assert self.max_prompt_size > tokens_in_text(self.system_prompt)

        assert self.cost_in_token > 0 and self.cost_out_token > 0
        self.dummy_response = " ".join(str(i) for i in range(53)) + "_"
        assert tokens_in_text(self.dummy_response) == self.avg_response_len
        self.reset()

    def add_to_context(self, user_message: str):
        ctx_user_msg = make_user_message(user_message)
        self.context.append(ctx_user_msg)
        self.context_tokens += context_token_len([ctx_user_msg])
        while self.context_tokens + self.expected_response_tokens > self.max_prompt_size:
            self.context_tokens -= context_token_len(self.context[1:2])
            self.context = self.context[:1] + self.context[2:]

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        self.add_to_context(user_message)
        self.costs_usd += self.cost_in_token * self.context_tokens
        self.costs_usd += self.cost_out_token * self.avg_response_len
        self.context.append(make_assistant_message(self.dummy_response))
        return self.dummy_response

    def reset(self):
        self.context = [make_system_message(self.system_prompt)]
        self.context_tokens = context_token_len(self.context)

    def save(self):
        pass
    
    def load(self):
        pass
