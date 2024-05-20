import os
import json
import time
from typing import Optional
from openai import OpenAI, ChatCompletion
from dataclasses import dataclass, field
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from model_interfaces.interface import ChatSession
from utils.llm import LLMContext, make_user_message, make_assistant_message


@dataclass
class HFChatSession(ChatSession):
    is_local: bool = True  # Costs are billed hourly. Hard to track.
    max_prompt_size: int = None
    model: str = None
    base_url: str = None
    context: LLMContext = field(default_factory=lambda: [])
    max_response_tokens: int = 2048
    client: OpenAI = None
    tokenizer: PreTrainedTokenizerFast = None
    tokens_used_last: int = 0

    @property
    def name(self):
        name = f"{super().name} - {self.model} - {self.max_prompt_size}"
        return name.replace("/", "-")

    def __post_init__(self):
        super().__post_init__()
        if self.base_url is None:
            self.base_url = os.getenv("HUGGINGFACE_API_BASE")
        assert self.base_url is not None
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1/",
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.removeprefix("huggingface/"))

    def try_twice(self) -> ChatCompletion:
        for i in range(2):
            try:
                return self.client.chat.completions.create(
                    model="tgi",
                    messages=self.context,
                    max_tokens=self.max_response_tokens,
                    temperature=0,
                )
            except Exception as exc:
                if i > 0:
                    raise exc
                time.sleep(3)

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        self.context.append(make_user_message(user_message))
        self.tokens_used_last += self.token_len(user_message) + self.max_response_tokens
        while self.tokens_used_last > self.max_prompt_size:
            self.tokens_used_last -= self.token_len(self.context[0]["content"])
            self.tokens_used_last -= self.token_len(self.context[1]["content"])
            self.context = self.context[2:]
        if agent_response is None:
            response = self.try_twice()
            self.tokens_used_last = response.usage.total_tokens
            response = response.choices[0].message.content.removesuffix("</s>")
        else:
            self.tokens_used_last -= self.max_response_tokens - self.token_len(agent_response)
            response = agent_response
        self.context.append(make_assistant_message(response))
        return response

    def reset(self):
        self.context = []

    def save(self):
        fname = self.save_path.joinpath("context.json")
        with open(fname, "w") as fd:
            json.dump(self.context, fd)

    def load(self):
        fname = self.save_path.joinpath("context.json")
        with open(fname, "r") as fd:
            self.context = json.load(fd)

    def token_len(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text=text))
