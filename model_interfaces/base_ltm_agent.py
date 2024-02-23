import abc
import codecs
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import List
from goodai.ltm.mem.base import RetrievedMemory

from model_interfaces.interface import ChatSession
from utils.openai import ask_llm
import tiktoken

_logger = logging.getLogger("exp_agent")
_log_prompts = os.environ.get("LTM_BENCH_PROMPT_LOGGING", "False").lower() in ["true", "yes", "1"]


class BaseLTMAgent(ChatSession, abc.ABC):
    """
    Abstract base of LTM agents
    """

    def __init__(self, run_name: str = "", model: str = None):
        super().__init__(run_name=run_name)
        self.model = model
        self.log_count = 0
        self.log_lock = threading.RLock()
        self.session_id = uuid.uuid4()

    @staticmethod
    def num_tokens_from_string(string: str, model="gpt-4"):
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))

    @classmethod
    def context_token_counts(cls, messages: List[dict]):
        """Calculates the total number of tokens in a list of messages."""
        total_tokens = 0
        for message in messages:
            total_tokens += cls.num_tokens_from_string(message["content"])
        return total_tokens

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

    def get_mem_excerpts(self, memories: List[RetrievedMemory], token_limit: int) -> str:
        token_count = 0
        excerpts: list[tuple[float, str]] = []
        ts = self.current_time
        for m in memories:
            ts_descriptor = self.get_elapsed_time_descriptor(m.timestamp, current_timestamp=ts)
            excerpt = f"## Excerpt from {ts_descriptor}\n{m.passage.strip()}\n\n"
            new_token_count = self.num_tokens_from_string(excerpt) + token_count
            if new_token_count > token_limit:
                break
            token_count = new_token_count
            excerpts.append(
                (
                    m.timestamp,
                    excerpt,
                )
            )
        excerpts.sort(key=lambda _t: _t[0])
        return "\n".join([e for _, e in excerpts])

    def completion(self, context: List[dict[str, str]], temperature: float, label: str) -> str:
        def cost_callback(cost_usd: float):
            self.costs_usd += cost_usd

        response = ask_llm(
            context, self.model, temperature=temperature, context_length=None,
            cost_callback=cost_callback
        )
        if _log_prompts:
            with self.log_lock:
                self.log_count += 1
                log_dir = f"./logs/{self.session_id}"
                os.makedirs(log_dir, exist_ok=True)
                prompt_file = f"{label}-prompt-{self.log_count}.json"
                prompt_json = json.dumps(context, indent=2)
                prompt_path = os.path.join(log_dir, prompt_file)
                with codecs.open(prompt_path, "w", "utf-8") as fd:
                    fd.write(prompt_json)
                completion_path = os.path.join(log_dir, f"{label}-completion-{self.log_count}.txt")
                with codecs.open(completion_path, "w", "utf-8") as fd:
                    fd.write(response)
        return response

    @abc.abstractmethod
    def reset_all(self):
        pass

    def reset(self):
        self.reset_all()

    @property
    @abc.abstractmethod
    def current_time(self) -> float:
        pass


@dataclass
class Message:
    role: str
    content: str
    timestamp: float

    def as_llm_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @property
    def is_user(self) -> bool:
        return self.role == "user"
