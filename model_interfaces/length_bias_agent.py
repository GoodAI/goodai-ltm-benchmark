import datetime
import json
import logging
import os
import time
from typing import List, Callable, Optional

from model_interfaces.base_ltm_agent import BaseLTMAgent, Message
from utils.json_utils import CustomEncoder
from utils.llm import make_system_message, make_user_message

_logger = logging.getLogger("exp_agent")
_log_prompts = os.environ.get('LTM_BENCH_PROMPT_LOGGING', 'False').lower() in ['true', 'yes', '1']
_default_system_message = """
You are a helpful AI assistant with a long-term memory. Prior interactions with the user are tagged with a timestamp. Current time: {datetime}.
"""


def _default_time(session_index: int, line_index: int) -> float:
    return time.time()


class LengthBiasAgent(BaseLTMAgent):
    """
    Control agent that biases retrieval based on the length of messages.
    """
    def __init__(self, max_prompt_size: int, time_fn: Callable[[int, int], float] = _default_time,
                 model: str = None, system_message: str = None, ctx_fraction_for_mem: float = 0.5,
                 llm_temperature: float = 0.01, run_name: str = ""):
        super().__init__(run_name=run_name, model=model)
        if system_message is None:
            system_message = _default_system_message
        self.llm_temperature = llm_temperature
        self.ctx_fraction_for_mem = ctx_fraction_for_mem
        self.max_prompt_size = max_prompt_size
        self.time_fn = time_fn
        self.session_index = 0
        self.system_message_template = system_message
        self.message_history: List[Message] = []

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size}"

    def build_llm_context(self, user_content: str) -> list[dict]:
        target_history_tokens = self.max_prompt_size * (1.0 - self.ctx_fraction_for_mem)
        context = []
        if self.system_message_template:
            context.append(make_system_message(
                self.system_message_template.format(datetime=datetime.datetime.now())))
        context.append(make_user_message(user_content))
        token_count = self.context_token_counts(context)
        removed_messages: list['Message'] = []
        for i in range(len(self.message_history) - 1, -1, -1):
            message = self.message_history[i]
            if message.is_user:
                et_descriptor = self.get_elapsed_time_descriptor(message.timestamp,
                                                                 self.current_time)
                new_content = f"[{et_descriptor}]\n{message.content}"
                message = Message(message.role, new_content, message.timestamp)
            message_dict = message.as_llm_dict()
            new_token_count = self.context_token_counts([message_dict]) + token_count
            if new_token_count > target_history_tokens:
                removed_messages = self.message_history[:i + 1]
                break
            context.insert(1, message_dict)
            token_count = new_token_count
        remain_tokens = self.max_prompt_size - token_count
        mem_message: dict = self.get_mem_message(removed_messages, remain_tokens)
        if mem_message:
            context.insert(1, mem_message)
        return context

    def get_mem_message(self, removed_messages: list['Message'],
                        remain_tokens: int) -> Optional[dict[str, str]]:
        excerpts_text = self.get_mocked_mem_excerpts(removed_messages, remain_tokens)
        excerpts_content = (f"The following are excerpts from the early part of the conversation "
                            f"or prior conversations, in chronological order:\n\n{excerpts_text}")
        return make_system_message(excerpts_content)

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

    def get_mocked_mem_excerpts(self, removed_messages: list['Message'], token_limit: int) -> str:
        removed_messages = sorted(removed_messages, key=lambda _m: len(_m.content))
        token_count = 0
        excerpts: list[tuple[float, str]] = []
        ts = self.current_time
        for m in removed_messages:
            if m.is_user:
                ts_descriptor = self.get_elapsed_time_descriptor(m.timestamp, current_timestamp=ts)
                excerpt = f"## Excerpt from {ts_descriptor}\n{m.role}:{m.content}\n\n"
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

    @property
    def current_time(self) -> float:
        return self.time_fn(self.session_index, len(self.message_history))

    def reply(self, user_content: str, agent_response: Optional[str] = None) -> str:
        context = self.build_llm_context(user_content)
        response = self.completion(context, temperature=self.llm_temperature, label="reply")
        user_message = Message(role='user', content=user_content, timestamp=self.current_time)
        self.message_history.append(user_message)
        assistant_message = Message(role='assistant', content=response, timestamp=self.current_time)
        self.message_history.append(assistant_message)
        return response

    def reset_history(self):
        self.message_history = []
        self.session_index += 1

    def reset_all(self):
        self.reset_history()
        self.session_index = 0

    def save(self):
        infos = [self.message_history]
        files = ["message_hist.json"]

        for obj, file in zip(infos, files):
            fname = self.save_path.joinpath(file)
            with open(fname, "w") as fd:
                json.dump(obj, fd, cls=CustomEncoder)

    def load(self):
        fname = self.save_path.joinpath("message_hist.json")
        with open(fname, "r") as fd:
            ctx = json.load(fd)
        message_hist = []
        for m in ctx:
            message_hist.append(Message(**m))
        self.message_history = message_hist
