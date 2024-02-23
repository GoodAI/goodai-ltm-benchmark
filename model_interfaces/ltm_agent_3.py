import datetime
import json
import logging
import re
import time
from typing import List, Callable, Optional
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.base import RetrievedMemory
from goodai.ltm.mem.config import ChunkExpansionConfig, TextMemoryConfig

from model_interfaces.base_ltm_agent import BaseLTMAgent, Message
from model_interfaces.exp_agents.prompts.scratchpad_ltm import s_ltm_template_queries_info
from utils.json_utils import CustomEncoder
from utils.openai import get_max_prompt_size

_logger = logging.getLogger("exp_agent")
_default_emb_model = "em-distilroberta-p3-01"
_txt_re = re.compile(r"^.*```(?:txt)?(.*)```.*$", re.MULTILINE | re.DOTALL)
_default_system_message = """
You are a helpful AI assistant with a long-term memory. Prior interactions with the user are tagged with a timestamp. Current time: {datetime}.
"""
_scratchpad_system_message = """
You are an expert in helping AI assistants manage their knowledge about a user and their 
operating environment.
"""


def _default_time(session_index: int, line_index: int) -> float:
    return time.time()


class LTMAgent3(BaseLTMAgent):
    """
    Uses a memory with semantic retrieval, chronological ordering of retrieved memory excerpts,
    and a user information scratchpad.
    """
    def __init__(
        self,
        max_prompt_size: int,
        time_fn: Callable[[int, int], float] = _default_time,
        system_message: str = None,
        ctx_fraction_for_mem: float = 0.5,
        model: str = None,
        emb_model: str = _default_emb_model,
        chunk_size: int = 32,
        overlap_threshold: float = 0.75,
        llm_temperature: float = 0.01,
        mem_temperature: float = 0.01,
        run_name: str = ""
    ):
        super().__init__(run_name=run_name, model=model)
        if system_message is None:
            system_message = _default_system_message
        self.mem_temperature = mem_temperature
        self.llm_temperature = llm_temperature
        self.overlap_threshold = overlap_threshold
        self.ctx_fraction_for_mem = ctx_fraction_for_mem
        self.max_prompt_size = min(max_prompt_size, get_max_prompt_size(model))
        self.time_fn = time_fn
        self.session_index = 0
        self.system_message_template = system_message
        self.message_history: List[Message] = []
        self.wm_scratchpad = ""
        mem_config = TextMemoryConfig()
        mem_config.queue_capacity = 50000
        mem_config.chunk_capacity = chunk_size
        mem_config.redundancy_overlap_threshold = overlap_threshold
        mem_config.chunk_expansion_config = ChunkExpansionConfig.for_line_break(
            min_extra_side_tokens=chunk_size, max_extra_side_tokens=chunk_size * 4
        )
        mem_config.reranking_k_factor = 10
        self.text_mem = AutoTextMemory.create(emb_model=emb_model, config=mem_config)

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size}"

    def build_llm_context(self, user_content: str) -> list[dict]:
        target_history_tokens = self.max_prompt_size * (1.0 - self.ctx_fraction_for_mem)
        new_system_content = ""
        if self.system_message_template:
            new_system_content = self.system_message_template.format(
                datetime=datetime.datetime.now())
        if self.wm_scratchpad:
            user_info_content = f"Current scratchpad content (world model, user info):\n" \
                                f"{self.wm_scratchpad}"
            if new_system_content:
                new_system_content = new_system_content + "\n\n" + user_info_content
            else:
                new_system_content = user_info_content
        context = []
        if new_system_content:
            context.append({"role": "system", "content": new_system_content})
        context.append({"role": "user", "content": user_content})
        token_count = self.context_token_counts(context)
        to_timestamp = self.current_time
        for message in reversed(self.message_history):
            if message.is_user:
                et_descriptor = self.get_elapsed_time_descriptor(message.timestamp, self.current_time)
                new_content = f"[{et_descriptor}]\n{message.content}"
                message = Message(message.role, new_content, message.timestamp)
            message_dict = message.as_llm_dict()
            new_token_count = self.context_token_counts([message_dict]) + token_count
            if new_token_count > target_history_tokens:
                break
            to_timestamp = message.timestamp
            context.insert(1, message_dict)
            token_count = new_token_count
        remain_tokens = self.max_prompt_size - token_count
        new_scratchpad, mem_message = self.get_mem_message(user_content, remain_tokens, to_timestamp=to_timestamp)
        if new_scratchpad is not None:
            self.wm_scratchpad = new_scratchpad
        if mem_message:
            context.insert(1, mem_message)
        return context

    def retrieve_from_queries(self, queries: list[str], k_per_query: int, to_timestamp: float) -> List[RetrievedMemory]:
        try:
            multi_list = self.text_mem.retrieve_multiple(queries, k=k_per_query)
        except IndexError:
            _logger.error(f"Unable to retrieve memories using these queries: {queries}")
            raise
        r_memories = [rm for entry in multi_list for rm in entry]
        r_memories = [rm for rm in r_memories if rm.timestamp < to_timestamp]
        r_memories.sort(key=lambda _rm: _rm.relevance, reverse=True)
        r_memories = RetrievedMemory.remove_overlaps(r_memories,
                                                     overlap_threshold=self.overlap_threshold)
        return r_memories

    def get_mem_message(
        self, user_content: str, remain_tokens: int, to_timestamp: float, k_per_query=250
    ) -> tuple[str, Optional[dict[str, str]]]:
        """
        Gets (1) a new user object, and (2) a context message with
        information from memory if available.
        """
        new_scratchpad = self.prepare_mem_info(user_content)
        queries = [f"user: {user_content}"]
        r_memories = self.retrieve_from_queries(queries, k_per_query=k_per_query, to_timestamp=to_timestamp)
        if not r_memories:
            return (
                new_scratchpad,
                None,
            )
        excerpts_text = self.get_mem_excerpts(r_memories, remain_tokens)
        excerpts_content = (
            f"The following are excerpts from the early part of the conversation "
            f"or prior conversations, in chronological order:\n\n{excerpts_text}"
        )
        return (
            new_scratchpad,
            dict(role="system", content=excerpts_content),
        )

    @staticmethod
    def _formatted_scratchpad(raw_scratchpad):
        return f"```txt\n{raw_scratchpad}\n```"

    @staticmethod
    def sanitize_and_parse_scratchpad(s_completion: str) -> Optional[str]:
        match_txt = _txt_re.search(s_completion)
        if match_txt:
            return match_txt.group(1)
        else:
            _logger.warning(f"Scratchpad content not found in completion: {s_completion}")
            return None

    def prepare_mem_info(self, user_content: str) -> str:
        prompt_messages = [{"role": "system", "content": _scratchpad_system_message}]
        last_assistant_message = None
        for i in range(len(self.message_history) - 1, -1, -1):
            m = self.message_history[i]
            if m.role == "assistant":
                last_assistant_message = m
                break
        if last_assistant_message:
            if len(self.message_history) > 2:
                prompt_messages.append({"role": "system", "content": "Prior conversation context omitted."})
            prompt_messages.append(last_assistant_message.as_llm_dict())
        if self.wm_scratchpad:
            user_info_description = f"Prior scratchpad content (world model, user info):\n" + self._formatted_scratchpad(
                self.wm_scratchpad
            )
        else:
            user_info_description = f"Prior scratchpad content is empty."
        sp_content = s_ltm_template_queries_info.format(
            user_info_description=user_info_description,
            user_content=user_content,
        ).strip()
        prompt_messages.append({"role": "user", "content": sp_content})
        s_completion = self.completion(prompt_messages, temperature=self.mem_temperature, label="scratchpad-generation")
        new_scratchpad = self.sanitize_and_parse_scratchpad(s_completion)
        return new_scratchpad

    @property
    def current_time(self) -> float:
        return self.time_fn(self.session_index, len(self.message_history))

    def add_to_memory(self, message: "Message"):
        text = f"{message.role}: {message.content}\n"
        self.text_mem.add_text(text, timestamp=message.timestamp)

    def reply(self, user_content: str) -> str:
        context = self.build_llm_context(user_content)
        response = self.completion(context, temperature=self.llm_temperature, label="reply")
        user_message = Message(role="user", content=user_content, timestamp=self.current_time)
        self.message_history.append(user_message)
        self.add_to_memory(user_message)
        assistant_message = Message(role="assistant", content=response, timestamp=self.current_time)
        self.message_history.append(assistant_message)
        self.add_to_memory(assistant_message)
        return response

    def reset_history(self):
        self.message_history = []
        self.session_index += 1

    def reset_all(self):
        self.reset_history()
        self.session_index = 0
        self.wm_scratchpad = ""
        self.text_mem.clear()
    def save(self):
        infos = [self.message_history, self.wm_scratchpad, self.text_mem.state_as_text()]
        files = ["message_hist.json", "scratchpad.json", "mem.json"]

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

        fname = self.save_path.joinpath("scratchpad.json")
        with open(fname, "r") as fd:
            self.wm_scratchpad = json.load(fd)

        fname = self.save_path.joinpath("mem.json")
        with open(fname, "r") as fd:
            self.text_mem.set_state(json.load(fd))
