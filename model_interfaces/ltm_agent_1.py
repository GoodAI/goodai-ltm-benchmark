import codecs
import datetime
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from json import JSONDecodeError
from typing import List, Callable, Optional

from goodai.helpers.json_helper import sanitize_and_parse_json
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.base import RetrievedMemory
from goodai.ltm.mem.config import ChunkExpansionConfig, TextMemoryConfig

from model_interfaces.exp_agents.prompts.chronological_ltm import cltm_template_queries_info
from model_interfaces.interface import ChatSession
from utils.constants import PERSISTENCE_DIR, ResetPolicy
from utils.json_utils import CustomEncoder
from utils.openai import ask_llm, get_max_prompt_size
import tiktoken

_logger = logging.getLogger("exp_agent")
_log_prompts = os.environ.get("LTM_BENCH_PROMPT_LOGGING", "False").lower() in ["true", "yes", "1"]
_default_system_message = """
You are a helpful AI assistant with a long-term memory. Prior interactions with the user are tagged with a timestamp. Current time: {datetime}.
"""
_user_info_system_message = """
You are an expert in helping AI assistants manage their knowledge about a user and their 
operating environment.
"""


def _default_time(session_index: int, line_index: int) -> float:
    return time.time()


class LTMAgent1(ChatSession):
    """
    Uses a memory with semantic retrieval, chronological ordering of retrieved memory excerpts,
    query generation, and a user information object in JSON format.
    """
    def __init__(
        self,
        max_prompt_size: int,
        time_fn: Callable[[int, int], float] = _default_time,
        system_message: str = None,
        ctx_fraction_for_mem: float = 0.5,
        model: str = None,
        emb_model: str = "st:sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 32,
        overlap_threshold: float = 0.75,
        llm_temperature: float = 0.01,
        mem_temperature: float = 0.01,
        run_name: str = ""
    ):
        super().__init__()
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
        self.user_info: dict = {}
        self.model = model
        self.session_id = uuid.uuid4()
        _logger.info(f"{LTMAgent1.__name__} session ID: {self.session_id}")
        self.log_lock = threading.RLock()
        self.log_count = 0
        self.run_name = run_name
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

    def build_llm_context(self, user_content: str) -> list[dict]:
        target_history_tokens = self.max_prompt_size * (1.0 - self.ctx_fraction_for_mem)
        new_system_content = ""
        if self.system_message_template:
            new_system_content = self.system_message_template.format(
                datetime=datetime.datetime.now())
        if self.user_info:
            user_info_text = json.dumps(self.user_info, indent=3)
            user_info_content = f"Current information about the user:\n{user_info_text}"
            if new_system_content:
                new_system_content = new_system_content + '\n\n' + user_info_content
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
                et_descriptor = self.get_elapsed_time_descriptor(message.timestamp,
                                                                 self.current_time)
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
        user_info, mem_message = self.get_mem_message(user_content, remain_tokens,
                                                      to_timestamp=to_timestamp)
        self.user_info = user_info
        if mem_message:
            context.insert(1, mem_message)
        return context

    def retrieve_from_queries(self, queries: list[str], k_per_query: int,
                              to_timestamp: float) -> List[RetrievedMemory]:
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
    ) -> tuple[dict, Optional[dict[str, str]]]:
        """
        Gets (1) a new user object, and (2) a context message with
        information from memory if available.
        """
        queries, user_info = self.prepare_mem_info(user_content)
        queries = queries or []
        queries = [f"user: {user_content}"] + queries
        r_memories = self.retrieve_from_queries(queries, k_per_query=k_per_query,
                                                to_timestamp=to_timestamp)
        if not r_memories:
            return (
                user_info,
                None,
            )
        excerpts_text = self.get_mem_excerpts(r_memories, remain_tokens)
        excerpts_content = (
            f"The following are excerpts from the early part of the conversation "
            f"or prior conversations, in chronological order:\n\n{excerpts_text}"
        )
        return (
            user_info,
            dict(role="system", content=excerpts_content),
        )

    def prepare_mem_info(self, user_content: str) -> tuple[list[str], dict]:
        prompt_messages = [{"role": "system", "content": _user_info_system_message}]
        last_assistant_message = None
        for i in range(len(self.message_history) - 1, -1, -1):
            m = self.message_history[i]
            if m.role == "assistant":
                last_assistant_message = m
                break
        if last_assistant_message:
            if len(self.message_history) > 2:
                prompt_messages.append({"role": "system",
                                        "content": "Prior conversation context omitted."})
            prompt_messages.append(last_assistant_message.as_llm_dict())
        if self.user_info:
            user_info_text = json.dumps(self.user_info, indent=3)
            user_info_description = f"Prior information about the user:\n{user_info_text}"
        else:
            user_info_description = f"There is no prior information about the user."
        sp_content = cltm_template_queries_info.format(
            user_info_description=user_info_description,
            user_content=user_content,
        ).strip()
        prompt_messages.append({"role": "user", "content": sp_content})
        query_json = self.completion(prompt_messages, temperature=self.mem_temperature,
                                     label="query-generation")
        try:
            queries_and_info = sanitize_and_parse_json(query_json)
        except (JSONDecodeError, ValueError):
            _logger.exception(f"Unable to parse JSON: {query_json}")
            queries_and_info = {}
        if not isinstance(queries_and_info, dict):
            _logger.warning("Query generation completion was not a dictionary!")
            queries_and_info = {}
        user_info = queries_and_info.get("user", self.user_info)
        _logger.info(f"New user object: {user_info}")
        queries = queries_and_info.get("queries", [])
        return (
            queries,
            user_info,
        )

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

    def reset_history(self):
        self.message_history = []
        self.session_index += 1

    def reset_all(self):
        self.reset_history()
        self.session_index = 0
        self.user_info = {}
        self.text_mem.clear()

    def reset(self):
        self.reset_all()

    def save(self):
        infos = [self.message_history, self.user_info, self.text_mem.state_as_text()]
        files = ["message_hist.json", "user_info.json", "mem.json"]

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

        fname = self.save_path.joinpath("user_info.json")
        with open(fname, "r") as fd:
            self.user_info = json.load(fd)

        fname = self.save_path.joinpath("mem.json")
        with open(fname, "r") as fd:
            self.text_mem.set_state(json.load(fd))

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
