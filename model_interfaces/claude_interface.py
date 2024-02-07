from dataclasses import dataclass, field

import anthropic

from model_interfaces.interface import ChatSession
from utils.openai import LLMContext, get_max_prompt_size, make_system_message, make_user_message, ensure_context_len, \
    make_assistant_message, token_cost


@dataclass
class ClaudeChatSession(ChatSession):

    system_prompt: str = "You are a helpful assistant."
    max_prompt_size: int = None
    model: str = "claude-2.1"
    verbose: bool = False
    context: LLMContext = field(default_factory=LLMContext)
    response_len: int = 1024

    def __post_init__(self):
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

        self.context, context_tokens = ensure_context_len(self.context, self.model, response_len=self.response_len)

        response = anthropic.Anthropic().beta.messages.create(
            model=self.model,
            max_tokens=self.response_len,
            messages=self.context)

        response_text = response.content[0].text

        if self.verbose:
            print(f"AGENT: {response_text}")
        self.context.append(make_assistant_message(response_text))

        price_in, price_out = token_cost(self.model)
        self.costs_usd += price_in * response.model_extra["usage"]["input_tokens"] + price_out * response.model_extra["usage"]["output_tokens"]

        return response_text

    def reset(self):
        self.context = []