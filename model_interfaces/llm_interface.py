import json
from dataclasses import dataclass, field
from datetime import datetime, time

import litellm

from model_interfaces.interface import ChatSession
from utils.json_utils import CustomEncoder
from utils.llm import LLMContext, make_user_message, make_assistant_message, make_system_message, get_model, \
    get_max_prompt_size, ask_llm


@dataclass
class LLMChatSession(ChatSession):
    system_prompt: str = "You are a helpful assistant."
    max_prompt_size: int = None
    model: str = None
    verbose: bool = False
    context: LLMContext = field(default_factory=LLMContext)
    max_response_tokens: int = 4096

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size}"

    def __post_init__(self):
        super().__post_init__()
        self.model = get_model(self.model)
        if self.max_prompt_size is None:
            self.max_prompt_size = get_max_prompt_size(self.model)
        else:
            self.max_prompt_size = min(self.max_prompt_size, get_max_prompt_size(self.model))

    def reply(self, user_message: str) -> str:
        if self.verbose:
            print(f"USER: {user_message}")

        def cost_callback(cost_usd: float):
            self.costs_usd += cost_usd

        self.context.append(make_user_message(user_message))
        response = ask_llm(self.context, self.model, context_length=self.max_prompt_size, cost_callback=cost_callback,
                           max_response_tokens=self.max_response_tokens)

        self.context.append(make_assistant_message(response))

        if self.verbose:
            print(f"AGENT: {response}")
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

    def token_len(self, text: str) -> int:
        return litellm.token_counter(self.model, text=text)


@dataclass
class TimestampLLMChatSession(LLMChatSession):
    system_prompt: str = "You are a helpful assistant. Prior interactions with the user are tagged with a timestamp."
    history: list = field(default_factory=list)

    @staticmethod
    def get_elapsed_time_descriptor(event_timestamp: float, current_timestamp: float):
        elapsed = current_timestamp - event_timestamp
        if elapsed < 1:
            return "just now"
        elif elapsed < 60:
            return f"{round(elapsed)} second(s) ago"
        elif elapsed < 60 * 60:
            return f"{round(elapsed / 60)} minute(s) ago"
        elif elapsed < 60 * 60 * 24:
            return f"{elapsed / (60 * 60):.1f} hour(s) ago"
        else:
            return f"{elapsed / (60 * 60 * 24):.1f} day(s) ago"

    def build_context(self, user_content: str) -> list[dict[str, str]]:
        def _ts_message(m: dict) -> dict:
            role = m["role"]
            new_content = m["content"]
            if role == "user":
                timestamp_dt = m["timestamp"]
                timestamp = timestamp_dt.timestamp()
                elapsed_desc = self.get_elapsed_time_descriptor(timestamp, time.time())
                new_content = f"[{elapsed_desc}]\n{new_content}"
            return {"role": role, "content": new_content}

        context = [make_system_message(self.system_prompt)]
        context.extend([_ts_message(m) for m in self.history])
        context.append(make_user_message(user_content))
        return context

    def reply(self, user_message: str) -> str:
        self.context = self.build_context(user_message)
        return super().reply(user_message)

    def reset(self):
        super().reset()
        self.history = []

    def save(self):
        fname = self.save_path.joinpath("history.json")
        with open(fname, "w") as fd:
            json.dump(self.history, fd, cls=CustomEncoder)

    def load(self):
        fname = self.save_path.joinpath("history.json")
        with open(fname, "r") as fd:
            self.history = json.load(fd)

        for h in self.history:
            h["timestamp"] = datetime.fromtimestamp(h["timestamp"])
