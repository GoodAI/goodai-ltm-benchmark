import datetime
import json
import logging
import uuid
import ltm.scratchpad as sp
from dataclasses import dataclass, field
from copy import deepcopy
from json import JSONDecodeError
from typing import List, Callable, Optional, Any, Union
from litellm import completion, completion_cost, token_counter
from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.agent import LTMAgentConfig, Message
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.base import RetrievedMemory
from goodai.ltm.mem.config import ChunkExpansionConfig, TextMemoryConfig

from utils.constants import DATA_DIR
from utils.llm import make_user_message, make_assistant_message, make_system_message, LLMContext

_debug_dir = DATA_DIR.joinpath("ltm_debug_info")
_logger = logging.getLogger("exp_agent")
_default_system_message = """
You are a helpful AI assistant with a long-term memory.
Prior interactions with the user are tagged with a timestamp.
Current time: {datetime}
Current information about the user:
{user_info}
""".strip()
_user_info_system_message = (
    "You are an expert in helping AI assistants manage their knowledge about a user and their operating environment."
)
_convo_excerpts_prefix = f"# The following are excerpts from the early part of the conversation "\
                         f"or prior conversations, in chronological order:\n\n"


def td_format(td: datetime.timedelta) -> str:
    seconds = int(td.total_seconds())
    periods = [
        ('year', 3600*24*365), ('month', 3600*24*30), ('day', 3600*24), ('hour', 3600), ('minute', 60), ('second', 1)
    ]
    parts = list()
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            parts.append("%s %s%s" % (period_value, period_name, has_s))
    if len(parts) == 0:
        return "just now"
    if len(parts) == 1:
        return f"{parts[0]} ago"
    return " and ".join([", ".join(parts[:-1])] + parts[-1:]) + " ago"


@dataclass
class LTMAgent:
    model: str
    max_prompt_size: int
    max_completion_tokens: Optional[int] = None
    config: LTMAgentConfig = field(default_factory=LTMAgentConfig)
    prompt_callback: Callable[[str, str, list[dict], str], Any] = None
    user_info: dict = field(default_factory=dict)
    user_info_ts: dict[str, float] = field(default_factory=dict)  # Tracks changes in user_info (timestamps)
    now: datetime.datetime = None  # Set in `reply` to keep a consistent "now" timestamp
    debug_level: int = 0
    llm_call_idx: int = None

    @property
    def save_name(self) -> str:
        return f"{self.model}-{self.max_prompt_size}-{self.max_completion_tokens}"

    def __post_init__(self):
        mem_config = TextMemoryConfig(
            queue_capacity=self.config.chunk_queue_capacity,
            chunk_capacity=self.config.chunk_size,
            chunk_overlap_fraction=self.config.chunk_overlap_fraction,
            redundancy_overlap_threshold=self.config.redundancy_overlap_threshold,
            chunk_expansion_config=ChunkExpansionConfig.for_line_break(
                min_extra_side_tokens=self.config.chunk_size // 4,
                max_extra_side_tokens=self.config.chunk_size * 4
            ),
            reranking_k_factor=10,
        )
        self.convo_mem = AutoTextMemory.create(emb_model=self.config.emb_model, config=mem_config)
        self.new_session()

    def new_session(self) -> 'LTMAgentSession':
        """
        Creates a new LTMAgentSession object and sets it as the current session.
        :return: The new session object.
        """
        session_id = str(uuid.uuid4())
        self.session = LTMAgentSession(session_id=session_id, m_history=[])
        if not self.convo_mem.is_empty():
            self.convo_mem.add_separator()
        return self.session

    def state_as_text(self) -> str:
        """
        :return: A string representation of the content of the agent's memories (including
        embeddings and chunks) in addition to agent configuration information.
        Note that callback functions are not part of the provided state string.
        """
        state = dict(
            model=self.model,
            max_prompt_size=self.max_prompt_size,
            max_completion_tokens=self.max_completion_tokens,
            config=self.config,
            convo_mem=self.convo_mem.state_as_text(),
            user_info=self.user_info,
            user_info_ts=self.user_info_ts,
            session=self.session.state_as_text(),
        )
        return json.dumps(state, cls=SimpleJSONEncoder)

    def from_state_text(self, state_text: str, prompt_callback: Callable[[str, str, list[dict], str], Any] = None):
        """
        Builds an LTMAgent given a state string previously obtained by
        calling the state_as_text() method.
        :param state_text: A string previously obtained by calling the state_as_text() method.
        :param prompt_callback: Optional function used to get information on prompts sent to the LLM.
        :return:
        """
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        self.max_prompt_size = state["max_prompt_size"]
        self.max_completion_tokens = state["max_completion_tokens"]
        self.model = state["model"]
        self.config = state["config"]
        self.user_info = state["user_info"]
        self.user_info_ts = state["user_info_ts"]
        self.prompt_callback = prompt_callback
        self.convo_mem.set_state(state["convo_mem"])
        self.session = LTMAgentSession.from_state_text(state["session"])

    def system_prompt(self) -> str:
        return _default_system_message.format(
            datetime=str(self.now)[:-7],
            user_info=json.dumps(self.user_info, indent=2),
        )

    def _build_llm_context(self, m_history: list[Message], user_content: str,
                           cost_callback: Callable[[float], Any]) -> list[dict]:
        target_history_tokens = self.max_prompt_size * (1.0 - self.config.ctx_fraction_for_mem)
        new_system_content = self.system_prompt()
        context = []
        if new_system_content:
            context.append(make_system_message(new_system_content))
        context.append(make_user_message(user_content))
        token_count = token_counter(model=self.model, messages=context)
        oldest_message_ts = self.now.timestamp()
        for message in reversed(m_history):
            if message.is_user:
                ts = datetime.datetime.fromtimestamp(message.timestamp)
                et_descriptor = f"{str(ts)[:-7]} ({td_format(self.now - ts)})"
                new_content = f"[{et_descriptor}]\n{message.content}"
                message = Message(message.role, new_content, message.timestamp)
            message_dict = message.as_llm_dict()
            new_token_count = token_counter(model=self.model, messages=[message_dict]) + token_count
            if new_token_count > target_history_tokens:
                break
            context.insert(1, message_dict)
            token_count = new_token_count
            oldest_message_ts = message.timestamp
        remaining_tokens = self.max_prompt_size - token_count
        self.user_info, mem_appendix = self._get_mem_message(
            m_history, user_content, remaining_tokens, oldest_message_ts, cost_callback,
        )
        if mem_appendix:
            context[0]["content"] += "\n\n" + mem_appendix
        return context

    def convo_retrieve(
        self, queries: list[str], k_per_query: int, before_timestamp: float,
    ) -> List[RetrievedMemory]:
        try:
            multi_list = self.convo_mem.retrieve_multiple(queries, k=k_per_query)
        except IndexError:
            _logger.error(f"Unable to retrieve memories using these queries: {queries}")
            raise
        r_memories = [rm for entry in multi_list for rm in entry if rm.timestamp < before_timestamp]
        r_memories = RetrievedMemory.remove_overlaps(
            r_memories, overlap_threshold=self.config.redundancy_overlap_threshold,
        )
        r_memories.sort(key=lambda _rm: _rm.timestamp)
        return r_memories

    def _get_mem_message(
        self, m_history: list[Message], user_content: str, remain_tokens: int,
        before_timestamp: float, cost_callback: Callable[[float], Any], k_per_query=250,
    ) -> tuple[Union[dict, str], str]:
        """
        Gets (1) a new user object, and (2) a context message with
        information from memory if available.
        """
        queries, user_info = self._prepare_mem_info(m_history, user_content,
                                                    cost_callback=cost_callback)
        queries = queries or []
        queries = [f"USER: {user_content}"] + queries
        convo_memories = self.convo_retrieve(queries, k_per_query, before_timestamp)
        excerpts_text = self.get_mem_excerpts(convo_memories, remain_tokens)
        return user_info, excerpts_text

    def _complete_prepare_user_info(self, prompt_messages: list[dict],
                                    user_content: str,
                                    cost_callback: Callable[[float], Any]) -> tuple[list[str], dict]:
        if self.user_info:
            user_info_text = json.dumps(self.user_info, indent=2)
            user_info_description = f"Prior information about the user:\n{user_info_text}"
        else:
            user_info_description = f"There is no prior information about the user."
        query_rewrite_msg = sp.query_rewrite_template.format(
            user_info_description=user_info_description,
            user_content=user_content,
        )
        prompt_messages.append({"role": "user", "content": query_rewrite_msg})
        query_json = self._completion(prompt_messages, temperature=self.config.mem_temperature,
                                      label="query-generation", cost_callback=cost_callback)
        try:
            queries_and_info = sanitize_and_parse_json(query_json)
        except (JSONDecodeError, ValueError):
            _logger.exception(f"Unable to parse JSON: {query_json}")
            queries_and_info = {}
        if not isinstance(queries_and_info, dict):
            _logger.warning("Query generation completion was not a dictionary!")
            queries_and_info = {}
        user_info = self._update_scratchpad(user_content, cost_callback)
        _logger.info(f"New user object: {user_info}")
        queries = queries_and_info.get("queries", [])
        return (
            queries,
            user_info,
        )

    def _prepare_mem_info(
            self, message_history: list[Message], user_content: str, cost_callback: Callable[[float], Any],
    ) -> tuple[list[str], Union[dict, str, None]]:
        prompt_messages = [make_system_message(_user_info_system_message)]
        messages_prefix = (
            "\n\nFor context, the assistant and the user have previously exchanged these messages:\n"
            "(prior conversation context omitted)\n\n"
        )
        token_count = token_counter(model=self.model, messages=prompt_messages)
        token_count += token_counter(model=self.model, text=messages_prefix)
        messages = list()
        token_limit = int(0.8 * self.max_prompt_size)
        for m in reversed(message_history[-10:]):
            msg = f"{m.role.upper()}: {m.content}"
            new_token_count = token_count + token_counter(model=self.model, text=msg)
            if new_token_count > token_limit:
                break
            token_count = new_token_count
            messages.append(msg)
        if len(messages) > 0:
            prompt_messages[0]["content"] += messages_prefix + "\n\n".join(reversed(messages))
        return self._complete_prepare_user_info(prompt_messages, user_content, cost_callback)

    def get_mem_excerpts(self, convo_memories: list[RetrievedMemory], token_limit: int) -> str:
        token_count = token_counter(model=self.model, text=_convo_excerpts_prefix)
        convo_excerpts: list[tuple[float, str]] = []
        for m in sorted(convo_memories, key=lambda _t: _t.relevance, reverse=True):
            ts = datetime.datetime.fromtimestamp(m.timestamp)
            ts_descriptor = f"{td_format(self.now - ts)} ({str(ts)[:-7]})"
            excerpt = f"# Excerpt from {ts_descriptor}\n{m.passage.strip()}\n\n"
            new_token_count = token_counter(model=self.model, text=excerpt) + token_count
            if new_token_count > token_limit:
                break
            token_count = new_token_count
            convo_excerpts.append((m.timestamp, excerpt))
        if convo_excerpts:
            convo_excerpts.sort(key=lambda _t: _t[0])
            return _convo_excerpts_prefix + "\n".join([e for _, e in convo_excerpts])
        return ""

    def _add_to_convo_memory(self, message: "Message"):
        role = "YOU" if message.role == "assistant" else message.role.upper()
        text = f"{role}: {message.content}\n"
        self.convo_mem.add_text(text, timestamp=message.timestamp)

    def reply(self, user_content: str, cost_callback: Callable[[float], Any] = None) -> str:
        """
        Asks the LLM to generate a completion from a user question/statement.
        This method first constructs a prompt from session history and memory excerpts.
        :param user_content: The user's question or statement.
        :param cost_callback: An optional function used to track LLM costs.
        :return: The agent's completion or reply.
        """
        self.now = datetime.datetime.now()
        session = self.session
        context = self._build_llm_context(session.message_history, user_content,
                                          cost_callback)
        response = self._completion(context, temperature=self.config.llm_temperature, label="reply",
                                    cost_callback=cost_callback)
        for role, content in [("user", user_content), ("assistant", response)]:
            message = Message(role=role, content=content, timestamp=self.now.timestamp())
            session.add(message)
            self._add_to_convo_memory(message)
        return response

    def _truncated_completion(self, context: LLMContext, max_messages: int, **kwargs) -> str:
        while len(context) + 1 > max_messages or token_counter(model=self.model,
                                                               messages=context) > self.max_prompt_size:
            context.pop(1)
        return self._completion(context, **kwargs)

    def _completion(self, context: List[dict[str, str]], temperature: float, label: str,
                    cost_callback: Callable[[float], Any]) -> str:
        # TODO: max_tokens is borked: https://github.com/BerriAI/litellm/issues/4439
        response = completion(model=self.model, messages=context, timeout=self.config.timeout,
                              temperature=temperature)
        response_text = response['choices'][0]['message']['content']
        if self.prompt_callback:
            self.prompt_callback(self.session.session_id, label, context, response_text)
        if cost_callback:
            cost = completion_cost(model=self.model, completion_response=response, messages=context)
            cost_callback(cost)
        self._debug_actions(context, temperature, response_text)
        return response_text

    def _debug_actions(self, context: list[dict[str, str]], temperature: float, response_text: str):
        if self.debug_level < 1:
            return

        # See if dir exists or create it, and set llm_call_idx
        save_dir = _debug_dir.joinpath(self.save_name)
        if self.llm_call_idx is None:
            if save_dir.exists() and len(list(save_dir.glob("*.txt"))) > 0:
                self.llm_call_idx = max(int(p.name.removesuffix(".txt")) for p in save_dir.glob("*.txt")) + 1
            else:
                self.llm_call_idx = 0
                save_dir.mkdir(parents=True, exist_ok=True)

        # Write content of LLM call to file
        save_path = save_dir.joinpath(f"{self.llm_call_idx:06d}.txt")
        with open(save_path, "w") as fd:
            fd.write(f"LLM temperature: {temperature}\n")
            for m in context:
                fd.write(f"--- {m['role'].upper()}\n{m['content']}\n")
            fd.write(f"--- Response:\n{response_text}")
        self.llm_call_idx += 1

        # Wait for confirmation
        if self.debug_level < 2:
            return
        print(f"LLM call saved as {save_path.name}")
        input("Press ENTER to continue...")

    def reset(self):
        self.convo_mem.clear()
        self.user_info = dict()
        self.new_session()

    def _update_scratchpad(
        self, user_message: str, cost_cb: Callable[[float], None],
        max_changes: int = 10, max_tries: int = 5, max_messages: int = 10, max_scratchpad_tokens: int = None,
    ) -> dict:
        """An embodied agent interacts with an extremely simple environment to safely update the scratchpad."""

        max_scratchpad_tokens = max_scratchpad_tokens or self.max_prompt_size // 4
        assert 0 < max_scratchpad_tokens < self.max_prompt_size // 2
        ask_kwargs = dict(label="scratchpad", temperature=0.01, cost_callback=cost_cb, max_messages=max_messages)
        scratchpad = deepcopy(self.user_info)
        scratchpad_timestamps = self.user_info_ts
        context = [make_system_message(sp.system_prompt_template.format(message=user_message))]
        for _ in range(max_changes):

            # This is how the info looks like, do you wanna make changes?
            context.append(make_user_message(sp.changes_yesno_template.format(user_info=sp.to_text(scratchpad))))
            response = self._truncated_completion(context, **ask_kwargs)
            if "yes" not in response.lower():
                break

            # Alright, make a single change
            changed = False
            context.append(make_assistant_message("yes"))
            context.append(make_user_message(sp.single_change_template))
            response = self._truncated_completion(context, **ask_kwargs)
            context.append(make_assistant_message(response))
            for _ in range(max_tries):
                try:
                    d = sp.extract_json_dict(response)
                    sp.add_new_content(scratchpad_timestamps, scratchpad, d["key"], d["new_content"])
                    changed = True
                    break
                except Exception as exc:
                    context.append(make_user_message(f"There has been an error:\n\n{str(exc)}\n\nTry again, but take this error into account."))
                    response = self._truncated_completion(context, **ask_kwargs)
                    context.append(make_assistant_message(response))

            # Register changes with timestamps and remove items if necessary
            if changed:
                keys_ts = None
                while token_counter(model=self.model, text=sp.to_text(scratchpad)) > max_scratchpad_tokens:
                    # Older and deeper items (with longer path) go first
                    if keys_ts is None:
                        keys_ts = sorted(
                            [(k, ts) for k, ts in scratchpad_timestamps.items()],
                            key=lambda t: (t[1], -len(t[0])),
                        )
                    keypath, _ = keys_ts.pop(0)
                    sp.remove_item(scratchpad, keypath)
                    del scratchpad_timestamps[keypath]
        return scratchpad


class LTMAgentSession:
    """
    An agent session, or a collection of messages.
    """
    def __init__(self, session_id: str, m_history: list[Message]):
        self.session_id = session_id
        self.message_history: list[Message] = m_history or []

    @property
    def message_count(self):
        return len(self.message_history)

    def state_as_text(self) -> str:
        """
        :return: A string that represents the contents of the session.
        """
        state = dict(session_id=self.session_id, history=self.message_history)
        return json.dumps(state, cls=SimpleJSONEncoder)

    def add(self, message: Message):
        self.message_history.append(message)

    @classmethod
    def from_state_text(cls, state_text: str) -> 'LTMAgentSession':
        """
        Builds a session object given state text.
        :param state_text: Text previously obtained using the state_as_text() method.
        :return: A session instance.
        """
        state: dict = json.loads(state_text, cls=SimpleJSONDecoder)
        session_id = state["session_id"]
        m_history = state["history"]
        return cls(session_id, m_history)
