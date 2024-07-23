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
from utils.text import truncate

_debug_dir = DATA_DIR.joinpath("ltm_debug_info")
_logger = logging.getLogger("exp_agent")
_default_system_message = """
You are a helpful AI assistant with a long-term memory.
Prior interactions with the user are tagged with a timestamp.
Current time: {datetime}
Current information about the user:
{user_info}
""".strip()


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
    debug_level: int = 1
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
        self.session = LTMAgentSession(session_id=session_id, m_history=[], i_dict={})
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

    def count_tokens(
        self, text: str | list[str] = None, messages: LLMContext = None, count_response_tokens: bool = False,
    ) -> int:
        return token_counter(model=self.model, text=text, messages=messages, count_response_tokens=count_response_tokens)

    def _build_llm_context(self, m_history: list[Message], user_content: str,
                           cost_callback: Callable[[float], Any]) -> list[dict]:
        target_history_tokens = self.max_prompt_size * (1.0 - self.config.ctx_fraction_for_mem)
        context = [
            make_system_message(_default_system_message.format(
                datetime=str(self.now)[:-7],
                user_info=sp.to_text(self.user_info),
            )),
            make_user_message(user_content),
        ]
        token_count = self.count_tokens(messages=context)
        oldest_message_ts = self.now.timestamp()
        for message in reversed(m_history):
            if message.is_user:
                ts = datetime.datetime.fromtimestamp(message.timestamp)
                et_descriptor = f"{str(ts)[:-7]} ({td_format(self.now - ts)})"
                new_content = f"[{et_descriptor}]\n{message.content}"
                message = Message(message.role, new_content, message.timestamp)
            message_dict = message.as_llm_dict()
            new_token_count = self.count_tokens(messages=[message_dict]) + token_count
            if new_token_count > target_history_tokens:
                break
            context.insert(1, message_dict)
            token_count = new_token_count
            oldest_message_ts = message.timestamp
        remaining_tokens = self.max_prompt_size - token_count
        self.user_info, recalled_memories = self._get_mem_message(
            m_history, user_content, remaining_tokens, oldest_message_ts, cost_callback,
        )
        for mem in recalled_memories[::-1]:
            context.insert(1, mem.as_llm_dict())

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
    ) -> tuple[Union[dict, str], List[Message]]:
        """
        Gets (1) a new user object, and (2) a context message with
        information from memory if available.
        """
        queries, user_info = self._prepare_mem_info(m_history, user_content,
                                                    cost_callback=cost_callback)
        queries = queries or []
        queries = [f"USER: {user_content}"] + queries
        convo_memories = self.convo_retrieve(queries, k_per_query, before_timestamp)
        interactions = self.get_mem_interactions(convo_memories, remain_tokens)
        return user_info, interactions

    def _last_messages_preview(self, message_history: list[Message], token_limit: int, max_messages: int = 10) -> str:
        messages = list()
        token_count = 0
        sep_tokens = self.count_tokens(text="\n\n")
        for m in reversed(message_history[-max_messages:]):
            msg = f"{m.role.upper()}: {truncate(m.content, 500)}"
            new_token_count = token_count + sep_tokens + self.count_tokens(text=msg)
            if new_token_count > token_limit:
                break
            token_count = new_token_count
            messages.append(msg)
        return "\n\n".join(reversed(messages))

    def _generate_extra_queries(
        self, message_history: list[Message], user_content: str, cost_callback: Callable[[float], Any],
    ) -> list[str]:
        # The system prompt includes this intro line, plus up to 10 previous messages in the conversation
        # TODO: Since user_info is presented in the user message, these past messages should be there too.
        system_prompt = ("You are an expert in helping AI assistants manage their knowledge about a user and their "
                         "operating environment.")
        messages_prefix = (
            "\n\nFor context, the assistant and the user have previously exchanged these messages:\n"
            "(prior conversation context omitted)\n\n"
        )
        system_tokens = self.count_tokens(text=system_prompt + messages_prefix)
        msg_preview = self._last_messages_preview(message_history, self.max_prompt_size - system_tokens)
        if msg_preview != "":
            system_prompt += messages_prefix + msg_preview

        if self.user_info:
            user_info_description = f"Prior information about the user:\n{sp.to_text(self.user_info)}"
        else:
            user_info_description = f"There is no prior information about the user."
        context = [
            make_system_message(system_prompt),
            make_user_message(sp.query_rewrite_template.format(
                user_info_description=user_info_description,
                user_content=user_content,
            )),
        ]
        query_json = self._completion(context, temperature=self.config.mem_temperature,
                                      label="query-generation", cost_callback=cost_callback)
        try:
            queries = sanitize_and_parse_json(query_json)
        except (JSONDecodeError, ValueError):
            _logger.exception(f"Unable to parse JSON: {query_json}")
            queries = {}
        if not isinstance(queries, dict):
            _logger.warning("Query generation completion was not a dictionary!")
            queries = {}
        return queries.get("queries", [])

    def _prepare_mem_info(
            self, message_history: list[Message], user_content: str, cost_callback: Callable[[float], Any],
    ) -> tuple[list[str], dict]:

        queries = self._generate_extra_queries(message_history, user_content, cost_callback)
        user_info = self._update_scratchpad(message_history, user_content, cost_callback)
        _logger.info(f"New user object: {user_info}")
        return queries, user_info

    def get_mem_interactions(self, convo_memories: list[RetrievedMemory], token_limit: int) -> list[Message]:
        token_count = 0
        relevant_interactions: list[Message] = []
        for m in sorted(convo_memories, key=lambda _t: _t.relevance, reverse=True):
            interaction_timestamp = m.timestamp
            interaction = deepcopy(self.session.interaction_from_timestamp(interaction_timestamp))

            new_token_count = self.count_tokens(messages=[i.as_llm_dict() for i in interaction]) + token_count
            if new_token_count > token_limit:
                break

            ts = datetime.datetime.fromtimestamp(interaction_timestamp)
            # Set the relative message timestamps
            ts_descriptor = f"{td_format(self.now - ts)} ({str(ts)[:-7]}) "
            interaction[0].content = ts_descriptor + interaction[0].content

            token_count = new_token_count
            relevant_interactions.extend(interaction)

        by_role = sorted(relevant_interactions, key=lambda a: a.role, reverse=True)
        by_timestamp = sorted(by_role, key=lambda a: a.timestamp)

        return by_timestamp

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

        interaction_timestamp = self.now.timestamp()
        user_message = Message(role="user", content=user_content, timestamp=interaction_timestamp)
        agent_message = Message(role="assistant", content=response, timestamp=interaction_timestamp)
        self._add_to_convo_memory(user_message)
        self._add_to_convo_memory(agent_message)
        self.session.add_interaction((user_message, agent_message))
        return response

    def _truncated_completion(self, context: LLMContext, max_messages: int, **kwargs) -> str:
        while len(context) + 1 > max_messages or self.count_tokens(messages=context) > self.max_prompt_size:
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
        self, message_history: list[Message], user_message: str, cost_cb: Callable[[float], None],
        max_changes: int = 10, max_tries: int = 5, max_messages: int = 10, max_scratchpad_tokens: int = 1024,
    ) -> dict:
        """An embodied agent interacts with an extremely simple environment to safely update the scratchpad.
        A rather small scratchpad (1k tokens) seems to work optimally."""
        assert 0 < max_scratchpad_tokens < self.max_prompt_size // 2
        ask_kwargs = dict(label="scratchpad", temperature=0, cost_callback=cost_cb, max_messages=max_messages)
        scratchpad = deepcopy(self.user_info)
        scratchpad_timestamps = self.user_info_ts
        preview_tokens = (self.max_prompt_size // 2) - self.count_tokens(text=sp.to_text(self.user_info))
        assert 0 < preview_tokens < self.max_prompt_size // 2
        msg_preview = self._last_messages_preview(message_history, preview_tokens)
        if msg_preview == "":
            msg_preview = "This assistant has not exchanged any messages with the user so far."
        else:
            msg_preview = (
                "The assistant has already exchanged some messages with the user:\n"
                f"(prior conversation context might be missing)\n\n{msg_preview}"
            )
        context = [make_system_message(sp.system_prompt_template.format(last_messages=msg_preview, message=user_message))]
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
            for _ in range(max_tries):
                response = self._truncated_completion(context, **ask_kwargs)
                context.append(make_assistant_message(response))
                if self.count_tokens(response) > 150:
                    context.append(make_user_message(
                        'This update is too large. Make it smaller or set "new_content" to null.'
                    ))
                    continue
                try:
                    d = sp.extract_json_dict(response)
                    if not d["new_content"]:
                        sp.remove_item(scratchpad, d["key"])
                    else:
                        sp.add_new_content(scratchpad_timestamps, scratchpad, d["key"], d["new_content"])
                        changed = True
                    break
                except Exception as exc:
                    context.append(make_user_message(
                        f"There has been an error:\n\n{str(exc)}\n\nTry again, but take this error into account."
                    ))

            # Register changes with timestamps and remove items if necessary
            if changed:
                keys_ts = None
                while self.count_tokens(text=sp.to_text(scratchpad)) > max_scratchpad_tokens:
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
    def __init__(self, session_id: str, m_history: list[Message], i_dict: dict[float, tuple[Message, Message]]):
        self.session_id = session_id
        self.message_history: list[Message] = m_history or []
        self.interaction_dict: dict[float, tuple[Message, Message]] = i_dict or {}

    @property
    def message_count(self):
        return len(self.message_history)

    def state_as_text(self) -> str:
        """
        :return: A string that represents the contents of the session.
        """
        state = dict(session_id=self.session_id, history=self.message_history, interactions=self.interaction_dict)
        return json.dumps(state, cls=SimpleJSONEncoder)

    def add(self, message: Message):
        self.message_history.append(message)

    def add_interaction(self, interaction: tuple[Message, Message]):
        self.message_history.extend(interaction)
        key = interaction[0].timestamp
        self.interaction_dict[key] = interaction

    def interaction_from_timestamp(self, timestamp: float) -> tuple[Message, Message]:
        return self.interaction_dict[timestamp]


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
        i_dict = state["interactions"]
        return cls(session_id, m_history, i_dict)