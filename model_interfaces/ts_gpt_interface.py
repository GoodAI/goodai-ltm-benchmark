import time

from model_interfaces.interface import ChatSession
from utils.openai import (
    ask_llm,
    LLMContext,
    make_system_message,
    make_user_message,
    make_assistant_message,
)
from dataclasses import dataclass, field


@dataclass
class TimestampGPTChatSession(ChatSession):
    system_prompt: str = "You are a helpful assistant. Prior interactions with the user are tagged with a timestamp."
    max_prompt_size: int = 8192
    model: str = "gpt-4"
    verbose: bool = False

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size}"

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
        context = self.build_context(user_message)
        if self.verbose:
            print(f"USER: {user_message}")

        def cost_callback(cost_usd: float):
            self.costs_usd += cost_usd

        response = ask_llm(
            context,
            model=self.model,
            temperature=0,
            context_length=self.max_prompt_size,
            cost_callback=cost_callback,
        )
        if self.verbose:
            print(f"AGENT: {response}")
        return response

    def reset(self):
        self.history = []

    def save(self):
        pass

    def load(self):
        pass
