import json
import uuid
from dataclasses import dataclass, field
import datetime
from math import ceil
from typing import Optional, Callable, Any, List, Tuple

from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.base import RetrievedMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory
from litellm import token_counter
from model_interfaces.base_ltm_agent import Message

from utils.llm import make_system_message, make_user_message, ask_llm, log_llm_call
from utils.text import td_format
from utils.ui import colour_print


class LTMAgentSession: #! deprecated -- to be replaced with hybrid mem
    """
    An agent session, or a collection of messages.
    """
    def __init__(self, session_id: str, m_history: list[Message], i_dict: dict[float, tuple[Message, Message]]):
        self.session_id = session_id
        self.message_history: list[tuple[Message, Message]] = m_history or []
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
        self.message_history.append(interaction)
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
        i_dict = {float(k): v for k, v in state["interactions"].items()}
        return cls(session_id, m_history, i_dict)


@dataclass
class InsertedContextAgent:
    max_completion_tokens: Optional[int] = None
    semantic_memory: DefaultTextMemory = None
    max_prompt_size: int = 16384
    is_local: bool = True
    defined_kws: list = field(default_factory=list)
    llm_call_idx: int = 0
    model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: float = 0.01
    system_message = """You are a helpful AI assistant."""
    debug_level: int = 1
    session: LTMAgentSession = None
    now: datetime.datetime = None  # Set in `reply` to keep a consistent "now" timestamp
    run_name: str = ""
    num_tries: int = 5

    @property
    def save_name(self) -> str:
        return f"{self.model.replace('/','-')}-{self.max_prompt_size}-{self.init_timestamp}"

    def __post_init__(self):
        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=50, chunk_overlap_fraction=0.0))
        self.max_message_size = 1000
        self.defined_kws = []
        self.new_session()
        self.init_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

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
        self.now = datetime.datetime.now()

        keywords = self.keywords_for_message(user_message, cost_cb=cost_callback)
        context = self.create_context(user_message, max_prompt_size=self.max_prompt_size, previous_interactions=0, cost_cb=cost_callback)
        response_text = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
        log_llm_call(self.run_name, self.save_name, self.debug_level, label=f"reply-{self.llm_call_idx }")
        self.llm_call_idx += 1

        # Save interaction to memory
        um = Message(role="user", content=user_message, timestamp=self.now.timestamp())
        am = Message(role="assistant", content=response_text, timestamp=self.now.timestamp())

        self.save_interaction(user_message, response_text, keywords)
        self.session.add_interaction((um, am))

        return response_text

    def keywords_for_message(self, user_message, cost_cb):

        prompt = 'Create two keywords to describe the topic of this message:\n"{user_message}".\n\nFocus on the topic and tone of the message. Produce the keywords in JSON like: `["keyword_1", "keyword_2"]`\n\nChoose keywords that would aid in retriving this message from memory in the future.\n\nReuse these keywords if appropriate: {keywords}'

        context = [make_system_message(prompt.format(user_message=user_message, keywords=self.defined_kws))]
        for _ in range(self.num_tries):
            try:
                print("Keyword gen")
                response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)

                keywords = [k.lower() for k in sanitize_and_parse_json(response)]
                break
            except Exception as e:
                print(repr(e) + response)
                continue

        # Update known list of keywords
        for k in keywords:
            if k not in self.defined_kws:
                self.defined_kws.append(k)

        print(f"Interaction keywords: {keywords}")
        return keywords

    def create_context(self, user_message, max_prompt_size, previous_interactions, cost_cb):

        context = [make_system_message(self.system_message), make_user_message(f"{str(self.now)[:-7]} ({td_format(datetime.timedelta(seconds=1))}) " + user_message)]
        relevant_interactions = self.get_relevant_memories(user_message, cost_cb)

        # Get interactions from the memories
        for m in relevant_interactions:
            if "trivia" in m[0].content:
                colour_print("YELLOW", f"<*** trivia ***>")
            else:
                colour_print("YELLOW", f"{m[0].content}")

        # Add the previous messages
        relevant_interactions = relevant_interactions + self.session.message_history[-previous_interactions:]

        # Add in memories up to the max prompt size
        current_size = token_counter(self.model, messages=context)
        shown_mems = 0
        target_size = max_prompt_size - self.max_message_size

        for interaction in reversed(relevant_interactions):
            user_interaction, assistant_interaction = interaction
            future_size = token_counter(self.model, messages=context + [user_interaction.as_llm_dict(), assistant_interaction.as_llm_dict()])

            # If this message is going to be too big, then skip it
            if shown_mems >= 100:
                break

            if future_size > target_size:
                continue

            # Add the interaction and count the tokens
            context.insert(1, assistant_interaction.as_llm_dict())

            ts = datetime.datetime.fromtimestamp(user_interaction.timestamp)
            et_descriptor = f"{str(ts)[:-7]} ({td_format(self.now - ts)}) "
            context.insert(1, user_interaction.as_llm_dict())
            context[1]["content"] = et_descriptor + context[1]["content"]

            shown_mems += 1

            current_size = future_size

        print(f"current context size: {current_size}")

        return context

    def llm_memory_filter(self, memories, queries, cost_cb):

        situation_prompt = """You are a part of an agent. Another part of the agent is currently searching for memories using the statements below.
Based on these statements, describe what is currently happening external to the agent in general terms:
{queries}  
"""

        prompt = """Here are a number of interactions, each is given a number:
{passages}         
*****************************************

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

        filtered_interactions = []

        # Get the situation
        queries_txt = "- " + "\n- ".join(queries)
        context = [make_user_message(situation_prompt.format(queries=queries_txt))]
        situation = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
        colour_print("MAGENTA", f"Filtering situation: {situation}")

        # Map retrieved memory fac
        interactions_to_filter, interaction_keywords = self.interactions_from_retrieved_memories(memories)

        num_splices = ceil(len(interactions_to_filter) / splice_length)
        # Iterate through the interactions_to_filter list and create the passage
        call_count = 0
        for splice in range(num_splices):
            start_idx = splice * splice_length
            end_idx = (splice + 1) * splice_length

            memories_passages = []
            memory_counter = 0

            for interaction, keywords in zip(interactions_to_filter[start_idx:end_idx], interaction_keywords[start_idx:end_idx]):
                um, am = interaction
                memories_passages.append(f"[MEMORY NUMBER {memory_counter} START].\n (User): {um.content}\n(You): {am.content}\nKeywords: {keywords}\n[MEMORY NUMBER {memory_counter} END]")
                memory_counter += 1

            passages = "\n\n------------------------\n\n".join(memories_passages)
            context = [make_user_message(prompt.format(passages=passages, situation=situation))]

            for _ in range(self.num_tries):
                try:
                    print("Attempting filter")
                    result = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                    log_llm_call(self.run_name, self.save_name, self.debug_level, label=f"reply-{self.llm_call_idx}-filter-{call_count}")

                    json_list = sanitize_and_parse_json(result)
                    for idx, selected_object in enumerate(json_list):
                        if selected_object["related"]:
                            filtered_interactions.append(interactions_to_filter[idx + start_idx])

                    call_count += 1
                    break
                except Exception as e:
                    print(e)
                    continue

        # print("Memories after LLM filtering")
        # for m in filtered_mems:
        #     colour_print("GREEN", m)

        return filtered_interactions

    def get_relevant_memories(self, user_message, cost_cb):
        prompt ="""Message from user: "{user_message}"
        
Given the above user question/statement, your task is to provide semantic queries and keywords for searching an archived 
conversation history that may be relevant to a reply to the user.

The search queries you produce should be compact reformulations of the user question/statement,
taking context into account. The purpose of the queries is accurate information retrieval. 
Search is purely semantic. 

Create a general query and a specific query. Pay attention to the situation and topic of the conversation including any characters or specifically named persons.
Use up to three of these keywords to help narrow the search:
{keywords}

The current time is: {time}. 

Write JSON in the following format:

{{
    "queries": array, // An array of strings: 2 descriptive search phrases, one general and one specific
    "keywords": array // An array of strings: 1 to 3 keywords that can be used to narrow the category of memories that are interesting. 
}}"""

        # Now create the context for generating the queries
        context = [make_user_message(prompt.format(user_message=user_message, time=self.now, keywords=self.defined_kws))]
        all_retrieved_memories = []
        query_keywords = []

        for _ in range(self.num_tries):
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
        for mem in keyword_filtered_mems[:10]:
            # print(f"Spreading with: {mem.passage}")
            for r_mem in self.semantic_memory.retrieve(mem.passage, k=5):
                if r_mem.relevance > 0.6:
                    keyword_filtered_mems.append(r_mem)

        # # TODO: Uncomment all this stuff when doing dev stuff
        # trivia_skip = False
        # for kw in all_keywords:
        #     if "trivia" in kw:
        #         trivia_skip = True
        #
        # if trivia_skip:
        #     llm_filtered_interactions, _ = self.interactions_from_retrieved_memories(keyword_filtered_mems)
        # else:
        #     llm_filtered_interactions = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], cost_cb)

        # TODO: ....And comment this one out
        llm_filtered_interactions = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], cost_cb)

        sorted_interactions = sorted(llm_filtered_interactions, key=lambda x: x[0].timestamp)
        return sorted_interactions

    def interactions_from_retrieved_memories(self, memory_chunks: List[RetrievedMemory]) -> Tuple[List[Tuple[Message, Message]], List[List[str]]]:
        interactions = []
        keywords = []
        for m in memory_chunks:
            interaction = self.session.interaction_from_timestamp(m.timestamp)
            if interaction not in interactions:
                interactions.append(interaction)
                keywords.append(m.metadata["keywords"])

        return interactions, keywords

    def save_interaction(self, user_message, response_message, keywords):
        self.semantic_memory.add_text(user_message, timestamp=self.now.timestamp(), metadata={"keywords": keywords})
        self.semantic_memory.add_separator()
        self.semantic_memory.add_text(response_message, timestamp=self.now.timestamp(), metadata={"keywords": keywords})
        self.semantic_memory.add_separator()

    def retrieve_from_keywords(self, keywords):
        selected_mems = []
        memories = self.semantic_memory.retrieve("", 2000)

        for m in memories:
            for kw in m.metadata["keywords"]:
                if kw in keywords:
                    selected_mems.append(m)
                    break

        return selected_mems

    def reset(self):
        self.semantic_memory.clear()
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
            session=self.session.state_as_text(),
            defined_kws=self.defined_kws,
            llm_call_idx=self.llm_call_idx
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
        self.semantic_memory.set_state(state["convo_mem"])
        self.session = LTMAgentSession.from_state_text(state["session"])
        self.defined_kws = state["defined_kws"]
        self.llm_call_idx = state["llm_call_idx"]










