import json
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
import datetime
from math import ceil
import unicodedata
from typing import Optional, Callable, Any, List, Tuple

from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.mem.base import RetrievedMemory
from litellm import token_counter

from ltm.memory.hybrid_memory import HybridMemory
from ltm.utils.config import Config
from model_interfaces.base_ltm_agent import Message

from utils.llm import make_system_message, make_user_message, ask_llm, log_llm_call, make_assistant_message
from utils.text import td_format
from utils.ui import colour_print


@dataclass
class InsertedContextAgent:  # ? worth adding most of this to the ltm.utils.config?
    max_completion_tokens: Optional[int] = None
    hybrid_memory: HybridMemory = None
    max_prompt_size: int = 16384
    is_local: bool = True
    defined_kws: list = field(default_factory=list)  # ? Switch to use sets?
    llm_call_idx: int = 0
    # model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    model: str = "gpt-4o-mini",
    temperature: float = 0.01
    system_message: str = "You are a helpful AI assistant."
    debug_level: int = 1
    session_id: Optional[str] = None
    now: datetime.datetime = None  # Set in `reply` to keep a consistent "now" timestamp
    run_name: str = ""
    num_tries: int = 5
    task_memory: list = field(default_factory=list)

    @property
    def save_name(self) -> str:
        sanitized_model = re.sub(r'[<>:"/\\|?*]', '_', self.model.replace('/', '-'))
        sanitized_timestamp = re.sub(r'[<>:"/\\|?*]', '_', self.init_timestamp.replace(':', '_'))
        return f"{sanitized_model}-{self.max_prompt_size}-{sanitized_timestamp}"

    def __post_init__(self):
        self.hybrid_memory = HybridMemory(Config.DATABASE_URL, Config.SEMANTIC_MEMORY_CONFIG,
                                          max_retrieve_capacity=2000)
        self.max_message_size = 1000  # ? Add to config?
        self.defined_kws = []  # ? Set?
        self.session_id = self.new_session()
        self.init_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    def save_interaction(self, user_message, response_message, keywords):
        self.hybrid_memory.add_interaction(self.session_id, user_message, response_message, self.now.timestamp(),
                                           keywords)

    def new_session(self) -> str:
        session_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().timestamp()
        self.hybrid_memory.create_session(session_id, created_at)
        if not self.hybrid_memory.is_empty():
            self.hybrid_memory.semantic_memory.add_separator()
        return session_id

    def reply(self, user_message: str, agent_response: Optional[str] = None, cost_callback: Callable = None) -> str:
        perform_task = """

Here are some additional tasks which you are to perform right now:
{task}

These tasks contains two things: a trigger condition, and a payload.
The trigger conditions have been met for all task. So they can be safely disregarded. Now all you need to do is perform the tasks described in the "payload" fields. 

You should append the result of your task to your reply to the query above

Append the results of your tasks to your current response.
"""

        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")
        self.now = datetime.datetime.now()

        keywords = self.keywords_for_message(user_message, cost_cb=cost_callback)

        if not agent_response:
            context = self.create_context(user_message, max_prompt_size=self.max_prompt_size, previous_interactions=0, cost_cb=cost_callback)

        else:
            context = [make_user_message(user_message)]

        # Update task memory
        self.add_task(deepcopy(context), cost_callback, user_message)

        # Get any tasks that might be due now.
        extra_tasks = self.progress_and_get_extra_tasks(context)
        if len(extra_tasks) > 0:
            context[-1]["content"] += perform_task.format(task=json.dumps(extra_tasks, indent=2))

        response_text = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
        # Sanitize the response text
        sanitized_response = self.sanitize_string(response_text)

        try:
            log_llm_call(self.run_name, self.save_name, self.debug_level, label=f"reply-{self.llm_call_idx}")
        except UnicodeEncodeError:
            print(f"Warning: Unable to log LLM call due to encoding issues. LLM call index: {self.llm_call_idx}")

        self.llm_call_idx += 1

        # Remove outdated tasks
        self.remove_completed_tasks()

        # Save interaction to memory
        self.hybrid_memory.add_interaction(self.session_id, user_message, sanitized_response, self.now.timestamp(),
                                           keywords)

        return sanitized_response

    @staticmethod
    def sanitize_string(s: str) -> str:
        """Remove or replace problematic characters."""
        return ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')

    def progress_and_get_extra_tasks(self, context):

        extra_tasks = []

        for task in self.task_memory:
            triggered = self.update_task(task, context)

            if triggered:
                extra_tasks.append(task["task_details"])

        return extra_tasks

    def add_task(self, context, cost_callback, user_message): # given some trigger condition
        task_should_add = """Here is the current list of future tasks:
{future_tasks}

----------
You are to decide if this message:
{user_message}
 
Contains instructions for a future task which is not already in the list of future tasks, and has very concrete conditions (either a number of messages, a specific date/time, or a 'trigger phrase').
- Does the description of the task imply that the thing that the agent has to do something given some future trigger?
- Does the user tell the agent that it should perform some kind of task in the future, along with some trigger condition for that task?
- Is the task already in the list?

Respond with your reasoning as to whether there has been a new future task discussed and with either "Should Update: Yes." or "Should Update: No." written exactly at the very end of your response."""

        new_task_def = """Create a new task in JSON format.
A task contains two things, a trigger condition and a payload.

Be very precise in how you define the task and record all relevant information about it including:
- The trigger condition should describe precisely and only the condition for the triggering of the task.
- The payload should describe precisely and only what should be done once the trigger condition is met.
- The trigger value should accurately state the value stated in the trigger condition.

Follow this format:
{
    "description": str, // A high level description of the task
    "task_details": {
        "trigger_condition": str // A precise description of the condition that should be met for the task to execute.
        "payload": str, // An exact description of what should be done when the trigger condition is met. Do not mention the trigger
        // Any other details that might be relevant
    }
    "trigger_details": {
        "trigger_condition": str, // Either MESSAGE_COUNT, MESSAGE_CONTENT, or TIME
        "trigger_value": str, // For MESSAGE_COUNT, the number of messages to wait. For TIME, a timestamp in the form `YYYY-mm-dd_HH:MM`. For MESSAGE_CONTENT, the exact text that will trigger the task.
    }
}"""

        context.append(make_user_message(task_should_add.format(future_tasks=json.dumps(self.task_memory, indent=2), user_message=user_message)))
        response_text = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
        log_llm_call(self.run_name, self.save_name, self.debug_level, label=f"should_update-{self.llm_call_idx}")
        colour_print("YELLOW", f"Should update: {response_text}")

        if "Should Update: No." in response_text:
            return
        elif "Should Update: Yes." not in response_text:
            return

        # Produce a new one
        context.append(make_assistant_message(response_text))
        context.append(make_user_message(new_task_def))

        try:
            response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
            log_llm_call(self.run_name, self.save_name, self.debug_level, label=f"do_update-{self.llm_call_idx}")

            new_task = sanitize_and_parse_json(response)
            self.process_new_task(new_task)

            print(f"Added Task:\n{json.dumps(new_task, indent=2)}\n")

        except Exception as e:
            print("Encountered exception when creating task: " + repr(e))

    def process_new_task(self, new_task: dict):
        new_task["trigger_details"]["triggered"] = False
        if new_task["trigger_details"]["trigger_condition"] == "MESSAGE_COUNT":
            new_task["trigger_details"]["trigger_value"] = int(new_task["trigger_details"]["trigger_value"])

        if new_task["trigger_details"]["trigger_condition"] == "MESSAGE_CONTENT":
            new_task["trigger_details"]["trigger_value"] = new_task["trigger_details"]["trigger_value"].lower()

        if new_task["trigger_details"]["trigger_condition"] == "TIME":
            new_task["trigger_details"]["trigger_value"] = time.mktime(datetime.datetime.strptime(new_task["trigger_details"]["trigger_value"], "%Y-%m-%d_%H:%M").timetuple())

        self.task_memory.append(new_task)

    def remove_completed_tasks(self):

        for task in self.task_memory:
            if task["trigger_details"]["triggered"]:
                self.task_memory.remove(task)

    def update_task(self, task, context):

        cond = task["trigger_details"]["trigger_condition"]
        val = task["trigger_details"]["trigger_value"]
        if cond == "MESSAGE_COUNT":
            # Decrement the message counter and ready it to be triggered
            task["trigger_details"]["trigger_value"] -= 1
            task["trigger_details"]["triggered"] = task["trigger_details"]["trigger_value"] <= 0

        elif cond == "MESSAGE_CONTENT":
            user_message = context[-1]["content"].lower()
            task["trigger_details"]["triggered"] = val in user_message

        elif cond == "TIME":
            task["trigger_details"]["triggered"] = self.now.timestamp() > val

        if task["trigger_details"]["triggered"]:
            print(f"{cond}: Task triggered: {json.dumps(task, indent=2)}")

        return task["trigger_details"]["triggered"]

    def keywords_for_message(self, user_message, cost_cb):

        prompt = """Create two keywords to describe the topic of this message:
        "{user_message}".
Focus on the topic and tone of the message. Produce the keywords in JSON like: `["keyword_1", "keyword_2"]`
Choose keywords that would aid in retrieving this message from memory in the future.
Reuse these keywords if appropriate: {keywords}"""

        context = [make_system_message(prompt.format(user_message=user_message, keywords=self.defined_kws))]
        keywords = []
        for _ in range(self.num_tries):
            try:
                print("Keyword gen")
                response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size,
                                   cost_callback=cost_cb, temperature=self.temperature)

                keywords = [k.lower() for k in sanitize_and_parse_json(response)]
                break
            except Exception as e:
                print(f"Keyword generation failed: {e}")
                continue

        # If keyword generation fails, use a default keyword
        if not keywords:
            keywords = ["general"]

        # Update known list of keywords
        for k in keywords:
            if k not in self.defined_kws:
                self.defined_kws.append(k)

        print(f"Interaction keywords: {keywords}")
        return keywords

    def create_context(self, user_message, max_prompt_size, previous_interactions, cost_cb):
        context = [make_system_message(self.system_message), make_user_message(
            f"{str(self.now)[:-7]} ({td_format(datetime.timedelta(seconds=1))}) " + user_message)]
        relevant_interactions = self.get_relevant_memories(user_message, cost_cb)

        # Get interactions from the memories
        for m in relevant_interactions:
            if "trivia" in m[0].content:
                colour_print("YELLOW", f"<*** trivia ***>")
            else:
                colour_print("YELLOW", f"{m[0].content}")

        # Add the previous messages
        recent_messages = self.hybrid_memory.get_recent_messages(self.session_id, limit=previous_interactions)
        relevant_interactions.extend([(Message(**msg.__dict__), Message(**msg.__dict__)) for msg in recent_messages])

        # Add in memories up to the max prompt size
        current_size = token_counter(self.model, messages=context)
        shown_mems = 0
        target_size = max_prompt_size - self.max_message_size

        for interaction in reversed(relevant_interactions):
            user_interaction, assistant_interaction = interaction
            future_size = token_counter(self.model, messages=context + [user_interaction.as_llm_dict(),
                                                                        assistant_interaction.as_llm_dict()])

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

        situation_prompt = """You are a part of an agent which is undergoing an exchange of messages with a user or multiple users.
Another part of the agent which is in direct contact with the user is currently searching for memories using the statements below in reaction to a message from the user.

Based on these statements, describe in general terms the topic of the current message. 
- Do not speculate on the reasons that the user may have for sending the message.
- Try to call out some specific details.

**********
STATEMENTS:
{queries}  
"""

        prompt = """Here are a number of interactions, each is given a number:
{passages}         
*****************************************
Each of these interactions might be related to the general situation below. Your task is to judge if these interactions have any relation to the general situation.
Filter out interactions that very clearly do not have any relation. But keep in interactions that have any kind of relationship to the situation such as in: topic, characters, locations, setting, etc.
- Does the topic of the interaction have any relationship to the situation?
- Does the interaction contain any of the details of the situation?

*****************************************
SITUATION:
{situation}

*****************************************
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
        situation = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb,
                            temperature=self.temperature)
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

            for interaction, keywords in zip(interactions_to_filter[start_idx:end_idx],
                                             interaction_keywords[start_idx:end_idx]):
                um, am = interaction
                memories_passages.append(
                    f"[MEMORY NUMBER {memory_counter} START].\n (User): {um.content}\n(You): {am.content}\nKeywords: {keywords}\n[MEMORY NUMBER {memory_counter} END]")
                memory_counter += 1

            passages = "\n\n------------------------\n\n".join(memories_passages)
            context = [make_user_message(prompt.format(passages=passages, situation=situation))]

            for _ in range(self.num_tries):
                try:
                    print("Attempting filter")
                    result = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size,
                                     cost_callback=cost_cb, temperature=self.temperature)
                    log_llm_call(self.run_name, self.save_name, self.debug_level,
                                 label=f"reply-{self.llm_call_idx}-filter-{call_count}")

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
        prompt = """Message from user: "{user_message}"

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
        context = [
            make_user_message(prompt.format(user_message=user_message, time=self.now, keywords=self.defined_kws))]
        all_retrieved_memories = []
        query_keywords = []

        for _ in range(self.num_tries):
            print("generating queries")
            response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size,
                               cost_callback=cost_cb, temperature=self.temperature)

            try:
                query_dict = sanitize_and_parse_json(response)
                query_keywords = [k.lower() for k in query_dict["keywords"]]
                print(f"Query keywords: {query_keywords}")

                all_retrieved_memories = []
                for q in query_dict["queries"] + [user_message]:
                    print(f"Querying with: {q}")
                    for mem in self.hybrid_memory.semantic_memory.retrieve(q, k=100):  # ? Add k to config?
                        all_retrieved_memories.append(mem)
                break
            except Exception as e:
                print(f"Query generation failed: {e}")
                continue

        all_keywords = query_keywords
        relevance_filtered_mems = [x for x in all_retrieved_memories if x.relevance > 0.6] + self.hybrid_memory.retrieve_from_keywords(all_keywords)  # ? Add relevance score to the config?

        relevance_filtered_mems.extend(self.hybrid_memory.retrieve_from_keywords(all_keywords))

        # Spreading activation
        print(f"Performing spreading activations with {len(relevance_filtered_mems[:10])} memories.")
        for mem in relevance_filtered_mems[:10]:
            for r_mem in self.hybrid_memory.semantic_memory.retrieve(mem.passage, k=5):
                if r_mem.relevance > 0.6:
                    relevance_filtered_mems.append(r_mem)

        llm_filtered_interactions = self.llm_memory_filter(relevance_filtered_mems, query_dict["queries"], cost_cb)

        sorted_interactions = sorted(llm_filtered_interactions, key=lambda x: x[0].timestamp)
        return sorted_interactions

    def interactions_from_retrieved_memories(self, memory_chunks: List[RetrievedMemory]) -> Tuple[
        List[Tuple[Message, Message]], List[List[str]]]:
        interactions = []
        keywords = []
        for m in memory_chunks:
            interaction = self.hybrid_memory.get_interaction_by_timestamp(self.session_id, m.timestamp)
            if interaction and interaction not in interactions:
                interactions.append(interaction)
                keywords.append(m.metadata.get("keywords", []))
        return interactions, keywords

    def reset(self):
        self.hybrid_memory.clear()
        self.new_session()

    def state_as_text(self) -> str:
        state = dict(
            model=self.model,
            max_prompt_size=self.max_prompt_size,
            max_completion_tokens=self.max_completion_tokens,
            hybrid_memory=self.hybrid_memory.state_as_text(),
            defined_kws=self.defined_kws,
            llm_call_idx=self.llm_call_idx
        )
        return json.dumps(state, cls=SimpleJSONEncoder)

    def from_state_text(self, state_text: str, prompt_callback: Callable[[str, str, list[dict], str], Any] = None):
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        self.max_prompt_size = state["max_prompt_size"]
        self.max_completion_tokens = state["max_completion_tokens"]
        self.model = state["model"]
        self.hybrid_memory.set_state(state["hybrid_memory"])
        self.defined_kws = state["defined_kws"]
        self.llm_call_idx = state["llm_call_idx"]