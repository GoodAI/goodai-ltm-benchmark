import json
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from typing import Optional, Callable, Any

from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory
from litellm import token_counter
import ltm.scratchpad as sp
from model_interfaces.base_ltm_agent import Message
from utils.constants import DATA_DIR

from utils.llm import LLMContext, make_system_message, make_user_message, make_assistant_message, ensure_context_len, \
    ask_llm
from utils.text import truncate
from utils.ui import colour_print

_debug_dir = DATA_DIR.joinpath("ltm_debug_info")


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

    def add_interaction(self, interaction: tuple[Message, Message]):
        self.message_history.extend(interaction)
        key = interaction[0].timestamp
        self.interaction_dict[key] = interaction

    def interaction_from_timestamp(self, timestamp: float) -> tuple[Message, Message]:
        return self.interaction_dict[timestamp]

    def by_index(self, idx):
        return self.message_history[idx]

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


@dataclass
class InsertedContextAgent:
    max_completion_tokens: Optional[int] = None
    semantic_memory: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    max_prompt_size: int = 16384
    is_local: bool = True
    defined_kws: list = field(default_factory=list)
    scratchpad: dict[str, str] = field(default_factory=dict)
    scratchpad_ts: dict[str, float] = field(default_factory=dict)  # Tracks changes in user_info (timestamps)
    llm_call_idx: int = 0
    costs_usd: float = 0.0
    model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: float = 0.01
    system_message = """You are a helpful AI assistant."""
    debug_level: int = 1
    session: LTMAgentSession = None

    @property
    def save_name(self) -> str:
        return f"{self.model}-{self.max_prompt_size}-{self.max_completion_tokens}__{self.init_timestamp}"

    def __post_init__(self):
        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=50, chunk_overlap_fraction=0.0))
        self.max_message_size = 1000
        self.defined_kws = []
        self.new_session()
        self.init_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    def new_session(self) -> 'LTMAgentSession':
        """
        Creates a new LTMAgentSession object and sets it as the current session.
        :return: The new session object.
        """
        session_id = str(uuid.uuid4())
        self.session = LTMAgentSession(session_id=session_id, m_history=[], i_dict={})
        if not self.semantic_memory.is_empty():
            self.semantic_memory.add_separator()
        return self.session

    def reply(self, user_message: str, agent_response: Optional[str] = None, cost_callback: Callable=None) -> str:

        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")

        keywords = self.keywords_for_message(user_message, cost_cb=cost_callback)
        context = self.create_context(user_message, max_prompt_size=self.max_prompt_size, previous_interactions=0, cost_cb=cost_callback)
        response_text = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
        debug_actions(context, self.temperature, response_text, self.llm_call_idx, self.debug_level, self.save_name,  name_template="call-{idx}")
        self.llm_call_idx += 1

        # Save interaction to memory
        user_ts = context[-1]["content"]
        timestamp_key = datetime.now().timestamp()
        um = Message(role="user", content=context[-1]["content"], timestamp=timestamp_key)
        am = Message(role="assistant", content=response_text, timestamp=timestamp_key)

        self.save_interaction(user_ts, response_text, timestamp_key, keywords)
        self.session.add_interaction((um, am))

        # Update scratchpad
        # self._update_scratchpad(self.session.message_history, user_message=user_message, cost_cb=cost_callback)
        return response_text

    def keywords_for_message(self, user_message, cost_cb):

        prompt = "Create two keywords to describe the topic of this message:\n'{user_message}'.\n\nFocus on the topic and tone of the message. Produce the keywords in JSON like: `[keyword_1, keyword_2]`\n\nChoose keywords that would aid in retriving this message from memory in the future.\n\nReuse these keywords if appropriate: {keywords}"

        context = [make_system_message(prompt.format(user_message=user_message, keywords=self.defined_kws))]
        while True:
            try:
                print("Keyword gen")
                response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)

                keywords = [k.lower() for k in sanitize_and_parse_json(response)]
                break
            except Exception as e:
                print(e)
                continue

        # Update known list of keywords
        for k in keywords:
            if k not in self.defined_kws:
                self.defined_kws.append(k)

        print(f"Interaction keywords: {keywords}")
        return keywords


    def create_context(self, user_message, max_prompt_size, previous_interactions, cost_cb):

        stamped_user_message = str(datetime.now()) + ": " + user_message
        context = [make_system_message(self.system_message.format(scratchpad=repr(self.scratchpad))), make_user_message(stamped_user_message)]
        relevant_memories = self.get_relevant_memories(user_message, cost_cb)

        # Get interactions from the memories
        full_interactions = []
        for m in relevant_memories:
            interaction = self.session.interaction_from_timestamp(m.timestamp)
            if interaction not in full_interactions:
                full_interactions.append(interaction)

        for m in full_interactions:
            if "trivia" in m[0].content:
                colour_print("YELLOW", f"<*** trivia ***>")
            else:
                colour_print("YELLOW", f"{m[0].content}")

        # Add the previous messages
        final_idx = self.session.message_count - 1
        while previous_interactions > 0 and final_idx > 0:

            # Agent reply
            context.insert(1, self.session.by_index(final_idx).as_llm_dict())
            # User message
            context.insert(1, self.session.by_index(final_idx-1).as_llm_dict())

            final_idx -= 2
            previous_interactions -= 1

        # Add in memories up to the max prompt size
        current_size = token_counter(self.model, messages=context)
        shown_mems = 0
        target_size = max_prompt_size - self.max_message_size

        for interaction in full_interactions[::-1]:
            user_message, assistant_message = interaction
            future_size = token_counter(self.model, messages=context + [user_message.as_llm_dict(), assistant_message.as_llm_dict()])

            # If this message is going to be too big, then skip it
            if shown_mems >= 100:
                break

            if future_size > target_size:
                continue

            # Add the interaction and count the tokens
            if user_message not in context:
                context.insert(1, assistant_message.as_llm_dict())
                context.insert(1, user_message.as_llm_dict())
                shown_mems += 1

                current_size = future_size

        print(f"current context size: {current_size}")

        return context

    def llm_memory_filter(self, memories, queries, keywords, cost_cb):

        situation_prompt = """You are a part of an agent. Another part of the agent is currently searching for memories using the statements below.
Based on these statements, describe what is currently happening external to the agent in general terms:
{queries}  
"""

        prompt = """Here are a number of interactions, each is given a number:
{passages}         
*********

Each of these interactions might be related to the general situation below. Your task is to judge if these interaction have any relation to the general situation.
Filter out interactions that very clearly do not have any relation. But keep in interactions that have any kind of relationship to the situation such as in: topic, characters, locations, setting, etc.

SITUATION:
{situation}

Express your answer in this JSON: 
[
    {{
        "number": int  // The number of the interaction.
        "justification": string  // Why the interaction is or is not related to the situation.
        "related": bool // Whether the interaction is related to the situation.
    }},
    ...
]
"""

        if len(memories) == 0:
            return []

        splice_length = 10

        added_timestamps = []
        mems_to_filter = []  # Memories without duplicates
        filtered_mems = []

        # Get the situation
        queries_txt = "- " + "\n- ".join(queries)
        context = [make_user_message(situation_prompt.format(queries=queries_txt))]
        situation = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
        colour_print("MAGENTA", f"Filtering situation: {situation}")

        # Remove memories that map to the same interaction
        for m in memories:
            timestamp = m.timestamp
            if timestamp not in added_timestamps:
                added_timestamps.append(timestamp)
                mems_to_filter.append(m)

        num_splices = ceil(len(mems_to_filter) / splice_length)
        # Iterate through the mems_to_filter list and create the passage
        call_count = 0
        for splice in range(num_splices):
            start_idx = splice * splice_length
            end_idx = (splice + 1) * splice_length

            memories_passages = []
            memory_counter = 0

            for m in mems_to_filter[start_idx:end_idx]:
                timestamp = m.timestamp
                um, am = self.session.interaction_from_timestamp(timestamp)
                memories_passages.append(f"{memory_counter}). (User): {um.content}\n(You): {am.content}\nKeywords: {m.metadata['keywords']}")
                memory_counter += 1

            queries_txt = "- " + "\n- ".join(queries)
            passages = "\n----\n".join(memories_passages)
            context = [make_user_message(prompt.format(passages=passages, situation=situation))]
            # context = [make_system_message(prompt.format(passages=passages, queries=queries_txt, keywords=keywords))]

            while True:
                try:
                    print("Attempting filter")
                    result = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                    debug_actions(context, self.temperature, result, self.llm_call_idx, self.debug_level, self.save_name, name_template="filter-{idx}-" + str(call_count))

                    json_list = sanitize_and_parse_json(result)
                    for idx, selected_object in enumerate(json_list):
                        if selected_object["related"]:
                            filtered_mems.append(mems_to_filter[idx + start_idx])

                    call_count += 1
                    break
                except Exception as e:
                    print(e)
                    continue

        filtered_mems = sorted(filtered_mems, key=lambda x: x.timestamp)
        # print("Memories after LLM filtering")
        # for m in filtered_mems:
        #     colour_print("GREEN", m)

        return filtered_mems

    def get_relevant_memories(self, user_message, cost_cb):
        prompt ="""Message from user: "{user_message}"
        
Given the above user question/statement, your task is to provide semantic queries and keywords for searching an archived 
conversation history that may be relevant to a reply to the user.

The search queries you produce should be compact reformulations of the user question/statement,
taking context into account. The purpose of the queries is accurate information retrieval. 
Search is purely semantic. 
g
Create a general query and a specific query. Pay attention to the situation and topic of the conversation including any characters or specifically named persons.
Use up to three of these keywords to help narrow the search:
{keywords}

The current time is {time}. If there is a time based query, use the time that you are interested in in the query.

Write JSON in the following format:

{{
    "queries": array, // An array of strings: 2 descriptive search phrases, one general and one specific
    "keywords": array // An array of strings: 1 to 3 keywords that can be used to narrow the category of memories that are interesting. 
}}"""

        # Now create the context for generating the queries
        context = [make_user_message(prompt.format(user_message=user_message, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), keywords=self.defined_kws))]
        all_retrieved_memories = []
        query_keywords = []
        while True:
            print("generating queries")
            response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)

            try:
                query_dict = sanitize_and_parse_json(response)
                query_keywords = [k.lower() for k in query_dict["keywords"]]
                print(f"Query keywords: {query_keywords}")

                all_retrieved_memories = []
                for q in query_dict["queries"] + [user_message]:
                    print(f"Querying with: {q}")
                    for mem in self.semantic_memory.retrieve(q, k=100):
                        if not self.memory_present(mem, all_retrieved_memories):
                            all_retrieved_memories.append(mem)
                break
            except Exception:
                continue

        # Filter by both relevance and keywords
        all_keywords = query_keywords
        relevance_filtered_mems = [x for x in all_retrieved_memories if x.relevance > 0.6] + self.retrieve_from_keywords(all_keywords)
        keyword_filtered_mems = []

        for m in relevance_filtered_mems:
            for kw in m.metadata["keywords"]:
                if kw in all_keywords:
                    keyword_filtered_mems.append(m)
                    break

        keyword_filtered_mems.extend(self.retrieve_from_keywords(all_keywords))

        # Spreading activations
        print(f"Performing spreading activations with {len(keyword_filtered_mems[:10])} memories.")
        secondary_memories = []
        for mem in keyword_filtered_mems[:10]:
            # print(f"Spreading with: {mem.passage}")
            for r_mem in self.semantic_memory.retrieve(mem.passage, k=5):
                if r_mem.relevance > 0.6 and not self.memory_present(r_mem, secondary_memories) and not self.memory_present(r_mem, keyword_filtered_mems):
                    secondary_memories.append(r_mem)

        keyword_filtered_mems.extend(secondary_memories)
        # # TODO: Trivia skip for speed
        # trivia_skip = False
        # for kw in all_keywords:
        #     if "trivia" in kw:
        #         trivia_skip = True
        #
        # if trivia_skip:
        #     llm_filtered_mems = keyword_filtered_mems
        # else:
        #     llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], all_keywords, cost_cb)

        llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], all_keywords, cost_cb)

        sorted_mems = sorted(llm_filtered_mems, key=lambda x: x.timestamp)
        return sorted_mems

    def memory_present(self, memory, memory_list):
        # passage_info seems to be unique to memory, regardless of the query
        for list_mem in memory_list:
            if memory.passage_info.fromIndex == list_mem.passage_info.fromIndex and memory.passage_info.toIndex == list_mem.passage_info.toIndex:
                return True
        return False

    def save_interaction(self, user_message, response_message, timestamp, keywords):
        self.semantic_memory.add_text(user_message, timestamp=timestamp, metadata={"keywords": keywords})
        self.semantic_memory.add_separator()
        self.semantic_memory.add_text(response_message, timestamp=timestamp, metadata={"keywords": keywords})
        self.semantic_memory.add_separator()

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

    def _update_scratchpad(
            self, message_history: list[Message], user_message: str, cost_cb: Callable[[float], None],
            max_changes: int = 10, max_tries: int = 5, max_messages: int = 10, max_scratchpad_tokens: int = 1024,
    ) -> dict:
        """An embodied agent interacts with an extremely simple environment to safely update the scratchpad.
        A rather small scratchpad (1k tokens) seems to work optimally."""
        assert 0 < max_scratchpad_tokens < self.max_prompt_size // 2
        ask_kwargs = dict(label="scratchpad", temperature=0, cost_callback=cost_cb, max_messages=max_messages)
        scratchpad = deepcopy(self.scratchpad)
        scratchpad_timestamps = self.scratchpad_ts
        preview_tokens = (self.max_prompt_size // 2) - self.count_tokens(text=sp.to_text(self.scratchpad))
        assert 0 < preview_tokens < self.max_prompt_size // 2
        msg_preview = self._last_messages_preview(message_history, preview_tokens)
        if msg_preview == "":
            msg_preview = "This assistant has not exchanged any messages with the user so far."
        else:
            msg_preview = (
                "The assistant has already exchanged some messages with the user:\n"
                f"(prior conversation context might be missing)\n\n{msg_preview}"
            )
        context = [
            make_system_message(sp.system_prompt_template.format(last_messages=msg_preview, message=user_message))]
        for _ in range(max_changes):

            # Perform analysis of the message and current scratchpad.
            context.append(make_user_message(sp.analysis_template.format(user_info=sp.to_text(scratchpad))))
            response = self._truncated_completion(context, **ask_kwargs)
            context.append(make_assistant_message(response))

            # This is how the info looks like, do you wanna make changes?
            context.append(make_user_message(sp.changes_yesno_template))
            response = self._truncated_completion(context, **ask_kwargs)
            if "yes" not in response.lower():
                break

            # Alright, make a single change
            changed = False
            context.append(make_assistant_message("yes"))
            context.append(make_user_message(sp.single_change_template))
            for _ in range(max_tries):
                response = self._truncated_completion(context, **ask_kwargs)
                print(f"Scratchpad what to change: {response}")
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

        self.scratchpad = scratchpad
        return scratchpad

    def retrieve_from_keywords(self, keywords):
        selected_mems = []
        memories = self.semantic_memory.retrieve("", 2000)

        for m in memories:
            for kw in m.metadata["keywords"]:
                if kw in keywords:
                    selected_mems.append(m)
                    break

        return selected_mems

    def _truncated_completion(self, context: LLMContext, max_messages: int, **kwargs) -> str:
        while len(context) + 1 > max_messages or self.count_tokens(messages=context) > self.max_prompt_size:
            context.pop(1)
        return ask_llm(context=context, model=self.model, cost_callback=kwargs["cost_callback"], temperature=self.temperature)

    def count_tokens(
        self, text: str | list[str] = None, messages: LLMContext = None, count_response_tokens: bool = False,
    ) -> int:
        return token_counter(model=self.model, text=text, messages=messages, count_response_tokens=count_response_tokens)

    def reset(self):
        self.semantic_memory.clear()
        self.scratchpad = dict()
        self.new_session()

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
            convo_mem=self.semantic_memory.state_as_text(),
            scratchpad=self.scratchpad,
            scratchpad_ts=self.scratchpad_ts,
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
        self.scratchpad = state["scratchpad"]
        self.scratchpad_ts = state["scratchpad_ts"]
        self.prompt_callback = prompt_callback
        self.semantic_memory.set_state(state["convo_mem"])
        self.session = LTMAgentSession.from_state_text(state["session"])


def debug_actions(context: list[dict[str, str]], temperature: float, response_text: str, llm_call_idx: int, debug_level: int, save_name: str, name_template: str = None):
    if debug_level < 1:
        return

    # See if dir exists or create it, and set llm_call_idx
    save_dir = _debug_dir.joinpath(save_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    if llm_call_idx is None:
        if save_dir.exists() and len(list(save_dir.glob("*.txt"))) > 0:
            llm_call_idx = max(int(p.name.removesuffix(".txt")) for p in save_dir.glob("*.txt")) + 1
        else:
            llm_call_idx = 0

    # Write content of LLM call to file
    if name_template:
        save_path = save_dir.joinpath(f"{name_template.format(idx=llm_call_idx)}.txt")
    else:
        save_path = save_dir.joinpath(f"{llm_call_idx:06d}.txt")

    with open(save_path, "w") as fd:
        fd.write(f"LLM temperature: {temperature}\n")
        for m in context:
            fd.write(f"--- {m['role'].upper()}\n{m['content']}\n")
        fd.write(f"--- Response:\n{response_text}")

    # Wait for confirmation
    if debug_level < 2:
        return
    print(f"LLM call saved as {save_path.name}")
    input("Press ENTER to continue...")







