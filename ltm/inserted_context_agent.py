import json
from dataclasses import dataclass, field
from datetime import datetime
from json import JSONDecodeError
from math import ceil
from typing import Optional

from goodai.helpers.json_helper import sanitize_and_parse_json
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory
from litellm import token_counter

from model_interfaces.interface import ChatSession
from utils.llm import LLMContext, make_system_message, make_user_message, make_assistant_message, ensure_context_len, \
    ask_llm
from utils.ui import colour_print


def dump_context_s(context):
    messages = []
    for message in context:
        messages.append(f"---\nRole: {message['role']}\nContent:\n{message['content']}")

    return "\n".join(messages)


def remove_timestamps(text: str):

    for idx, c in enumerate(text):
        if c.lower() in "abcdefghijklmnopqrstuvwxyz[](){}":
            return text[idx:]
    return text


class MessageArchive:

    def __init__(self):
        self.message_list = []
        self.message_dict = {}

    def add_interaction(self, key, interaction):
        self.message_list.extend(interaction)
        self.message_dict[key] = interaction

    def get_from_key(self, key):
        return self.message_dict[key]

    def length(self):
        return len(self.message_list)

    def by_index(self, index):
        return self.message_list[index]


@dataclass
class InsertedContextAgent(ChatSession):
    all_messages: MessageArchive = field(default_factory=MessageArchive)
    semantic_memory: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    max_prompt_size: int = 16384
    is_local: bool = True
    defined_kws: set = field(default_factory=set)
    scratchpad: dict[str, str] = field(default_factory=dict)
    llm_index: int = 0

    system_message = """
You have previously recorded information over your lifetime. This data is an aggregated report of most of the messages up until now:
{scratchpad}     
"""

    should_update = """{user_info_description}

== New user interaction ==
{user_content}
==

Based on prior user information and the above interaction with the user, your task is to decide carefully if you should
update the scratchpad, and what changes should be made. Ignore all data that looks like general trivia.

Try to answer these questions:
- Does the interaction contain information that will be useful in future?
- Is the information unimportant general knowledge, or useful user specific knowledge?

Sketch out some general plan for the data you think should be written to the scratchpad.

Write JSON in the following format:

{{
    "reasoning": string, // Does the user query/statement contain information relating to the user or something they may expect you to keep track of?
    "verdict": bool // The decision of whether to write something to the scratchpad.  
}}"""

    update_user_info = """
Based on prior user information and the above interaction with the user, your task is to provide 
a new user object with updated information provided by the user, such as 
facts about themselves or information they are expecting you to keep track of.
Consider carefully if the user implicitly or explicitly wishes for you to save the information.

The updated user object should be compact. Avoid storing unimportant general knowledge.
At the same time, it's important to preserve prior information 
you're keeping track of for the user. Capture information provided by the user without
omitting important details. Exercise judgment in determining if new information overrides, 
deletes or augments existing information. Property names should be descriptive.

Write JSON in the following format:

{{
    "user": {{ ... }}, // An updated user object containing attributes, facts, world models
}}"""

    def __post_init__(self):
        super().__post_init__()

        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=50, chunk_overlap_fraction=0.0))
        self.is_local = True
        self.max_message_size = 1000

    def add_cost(self, cost):
        self.costs_usd += cost

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:

        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")
        context = self.create_context(user_message, max_prompt_size=self.max_prompt_size, previous_interactions=5)

        fname = f"data/llm_calls/call_{self.llm_index}.json"
        with open(fname, "w") as f:
            colour_print("BlUE", f"Saving to: {fname}")
            f.write(dump_context_s(context))
            response_text = ask_llm(context, model="gpt-4o", max_overall_tokens=self.max_prompt_size,
                                    cost_callback=self.add_cost)

            response_text = remove_timestamps(response_text)

            f.write(f"\n\nResponse:\n{response_text}")
            self.llm_index += 1

        response_ts = str(datetime.now()) + ": " + response_text

        timestamp_key = datetime.now()
        self.all_messages.add_interaction(timestamp_key, [context[-1], make_assistant_message(response_ts)])

        # Save interaction to memory
        user_ts = context[-1]["content"]
        self.save_interaction(user_ts, response_ts, timestamp_key)

        # Update scratchpad
        interaction = "(User) " + user_ts + "\n---\n" + "(Agent) " + response_ts
        self.update_scratchpad(interaction)

        colour_print("Magenta", f"Current total cost: {self.costs_usd}")
        return response_text

    def create_context(self, user_message, max_prompt_size, previous_interactions):

        stamped_user_message = str(datetime.now()) + ": " + user_message
        context = [make_system_message(self.system_message.format(scratchpad=repr(self.scratchpad))), make_user_message(stamped_user_message)]
        relevant_memories = self.get_relevant_memories(user_message)

        for m in relevant_memories:
            colour_print("YELLOW", f"Got memory: {m}")

        # Add the previous messages
        final_idx = self.all_messages.length() - 1
        while previous_interactions > 0 and final_idx > 0:

            # Agent reply
            context.insert(1, self.all_messages.by_index(final_idx))
            # User message
            context.insert(1, self.all_messages.by_index(final_idx-1))

            final_idx -= 2
            previous_interactions -= 1

        # Add in memories up to the max prompt size
        current_size = token_counter("gpt-4o", messages=context)
        memory_idx = len(relevant_memories) - 1
        max_mems_to_show = 10

        while current_size < max_prompt_size - self.max_message_size and memory_idx >= 0 and max_mems_to_show > 0:
            user_message, assistant_message = self.messages_from_memory(relevant_memories[memory_idx])

            if user_message not in context:
                context.insert(1, assistant_message)
                context.insert(1, user_message)
                max_mems_to_show -= 1

                current_size = token_counter("gpt-4o", messages=context)
            memory_idx -= 1

        print(f"current context size: {current_size}")

        return context

    # def get_relevant_memories(self, user_message):
    #     relevant_memories = self.semantic_memory.retrieve(user_message, k=20)
    #     return sorted([x for x in relevant_memories if x.relevance > 0.7], key=lambda x: x.timestamp)

    def llm_memory_filter(self, memories, queries):
        prompt = """Here are a number of passages, each is given a number:
{passages}         
        
Indicate which passages have some relation to any of the following queries. Even a loose connection is okay, but it should be justifiable:
{queries}

Express your answer as a JSON list. For example: `[2, 3, ..., 7]`
"""
        if len(memories) == 0:
            return []

        splice_length = 10

        added_timestamps = []
        mems_to_filter = []  # Memories without duplicates
        filtered_mems = []

        # Remove memories that map to the same interaction
        for m in memories:
            timestamp = m.metadata["timestamp"]
            if timestamp not in added_timestamps:
                added_timestamps.append(timestamp)
                mems_to_filter.append(m)

        num_splices = ceil(len(mems_to_filter) / splice_length)
        # Iterate through the mems_to_filter list and create the passage
        for splice in range(num_splices):
            start_idx = splice * splice_length
            end_idx = (splice + 1) * splice_length

            memories_passages = []
            memory_counter = 0

            for m in mems_to_filter[start_idx:end_idx]:
                timestamp = m.metadata["timestamp"]
                um, am = self.all_messages.get_from_key(timestamp)
                memories_passages.append(f"{memory_counter}). (User): {um['content']}\n(You): {am['content']}")
                memory_counter += 1

            passages = "\n----\n".join(memories_passages)
            context = [make_system_message(prompt.format(passages=passages, queries="\n".join(queries)))]

            while True:
                try:
                    print("Attempting filter")
                    result = ask_llm(context, model="gpt-4o", max_overall_tokens=16384, cost_callback=self.add_cost)

                    json_list = sanitize_and_parse_json(result)
                    for idx in json_list:
                        filtered_mems.append(mems_to_filter[idx + start_idx])
                    break
                except:
                    continue

        print("Memories after LLM filtering")
        for m in filtered_mems:
            colour_print("GREEN", m)

        return filtered_mems

    def get_relevant_memories(self, user_message):
        prompt ="""Message from user: "{user_message}"
        
Given the above user question/statement, your task is to provide semantic queries for searching archived 
conversation history that may be relevant to a reply to the user.

The search queries you produce should be compact reformulations of the user question/statement,
taking context into account. The purpose of the queries is accurate information retrieval. 
Search is purely semantic. 

The current time is {time}. If there is a time based query, use the time that you are interested in in the query.

Write JSON in the following format:

{{
    "queries": array, // An array of strings: 1 or 2 descriptive search phrases
}}"""

        context = [make_user_message(prompt.format(user_message=user_message, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))]
        all_retrieved_memories = []
        while True:
            response = ask_llm(context, model="gpt-4o", max_overall_tokens=self.max_prompt_size, cost_callback=self.add_cost)

            try:
                query_dict = sanitize_and_parse_json(response)

                all_retrieved_memories = []
                for q in query_dict["queries"] + [user_message]:
                    print(f"Querying with: {q}")
                    for mem in self.semantic_memory.retrieve(q, k=20):
                        if not self.memory_present(mem, all_retrieved_memories):
                            all_retrieved_memories.append(mem)

                break
            except Exception:
                continue

        colour_print("MAGENTA", "\nMEMORIES BEFORE FILTERING:")
        for m in all_retrieved_memories:
            colour_print("MAGENTA", m)

        filtered_mems = self.llm_memory_filter([x for x in all_retrieved_memories if x.relevance > 0.6], query_dict["queries"])

        # Conditional Spreading activations
        if 0 < len(filtered_mems) < 10:
            print(f"Performing spreading activations with {len(filtered_mems)} memories.")
            secondary_memories = []
            for first_mem in filtered_mems:
                print(f"Spreading with: {first_mem.passage}")
                for mem in self.semantic_memory.retrieve(first_mem.passage, k=5):
                    if mem.relevance > 0.6 and not self.memory_present(mem, secondary_memories) and not self.memory_present(mem, filtered_mems):
                        secondary_memories.append(mem)

            filtered_mems.extend(secondary_memories)
            # filtered_mems.extend(self.llm_memory_filter(secondary_memories, query_dict["queries"]))

        sorted_mems = sorted(filtered_mems, key=lambda x: x.timestamp)
        return sorted_mems

    def memory_present(self, memory, memory_list):
        # passage_info seems to be unique to memory, regardless of the query
        for list_mem in memory_list:
            if memory.passage_info.fromIndex == list_mem.passage_info.fromIndex and memory.passage_info.toIndex == list_mem.passage_info.toIndex:
                return True
        return False

    def save_interaction(self, user_message, response_message, timestamp):
        self.semantic_memory.add_text(user_message, metadata={"timestamp": timestamp})
        self.semantic_memory.add_separator()
        self.semantic_memory.add_text(response_message, metadata={"timestamp": timestamp})
        self.semantic_memory.add_separator()

    def messages_from_memory(self, memory):
        timestamp_key = memory.metadata["timestamp"]

        interaction = self.all_messages.get_from_key(timestamp_key)
        return interaction[0], interaction[1]

    def update_scratchpad(self, instruction):

        scratchpad_text = json.dumps(self.scratchpad, indent=2)

        # colour_print("lightblue", f"Updating old scratchpad: {scratchpad_text}")

        context = [make_user_message(
            self.should_update.format(user_info_description=scratchpad_text, user_content=instruction).strip())]

        _, size = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
        print(f"Scratchpad change decide context size: {size}")
        decision_json = ask_llm(model="gpt-4o", context=context, cost_callback=self.add_cost, max_overall_tokens=16384)


        # colour_print("yellow", f"\nDecision to update scratchpad: {json.dumps(decision_json, indent=2)}")

        # Parse out decision whether to update the
        try:
            decision = sanitize_and_parse_json(decision_json)
        except (JSONDecodeError, ValueError):
            colour_print("RED", f"Unable to parse JSON: {decision_json}")
            decision = {}
        if not isinstance(decision, dict):
            colour_print("RED", "Query generation completion was not a dictionary!")
            decision = {}

        if decision.get("verdict", True):

            context.append(make_assistant_message(decision_json))
            context.append(make_user_message(self.update_user_info.format(user_info_description=scratchpad_text, user_content=instruction).strip()))
            _, size = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
            print(f"Scratchpad Perform Change context size: {size}")
            scratchpad_json = ask_llm(model="gpt-4-turbo", context=context, cost_callback=self.add_cost, max_overall_tokens=16384)

            try:
                pad = sanitize_and_parse_json(scratchpad_json)
            except (JSONDecodeError, ValueError):
                colour_print("RED", f"Unable to parse JSON: {scratchpad_json}")
                pad = {}
            if not isinstance(pad, dict):
                colour_print("RED", "Query generation completion was not a dictionary!")
                pad = {}

            self.scratchpad = pad.get("user", self.scratchpad)
            # colour_print("lightblue", f"\n\nNEW scratchpad: {json.dumps(self.scratchpad, indent=2)}")

    def reset(self):
        pass

    def save(self):
        pass

    def load(self):
        pass




