import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from json import JSONDecodeError
from typing import Optional, Dict, Callable, List

import litellm
import pystache
from goodai.helpers.json_helper import sanitize_and_parse_json
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory

from model_interfaces.interface import ChatSession
from utils.llm import LLMContext, make_system_message, make_user_message, make_assistant_message, ask_llm, \
    ensure_context_len
from utils.ui import colour_print

# litellm.set_verbose=True


def use_tools(functions, tool_use):
    try:
        returned_context = []
        for tool in tool_use:
            fun = functions[tool.function.name]
            args = json.loads(tool.function.arguments)

            print(f"\tCalling '{tool.function.name}' with args {args} and id: {tool.id}")
            result = fun(**args)
            print(f"\t\tReturning function '{tool.function.name}' with id: {tool.id}")
            returned_context.append({
                "tool_call_id": tool.id,
                "role": "tool",
                "name": tool.function.name,
                "content": result,
            })
        return True, returned_context
    except Exception as e:
        print(e)
        return False, []


def dump_context(context):
    for message in context:
        colour_print("YELLOW", f"{message['role']}: {message['content']}\n")


@dataclass
class LTMAgentV1(ChatSession):
    context: LLMContext = field(default_factory=list)
    functions: Dict[str, Callable] = None
    inner_loop_responses: List[str] = field(default_factory=list)
    loop_active: bool = False
    semantic_memory: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    max_prompt_size: int = 16384
    is_local: bool = True
    defined_kws: set = field(default_factory=set)
    scratchpad: dict[str, str] = field(default_factory=dict)

    inner_loop_system_prompt: str = """You are an assistant for a user. Your interaction with the user is like a game and will operate in an outer-loop and an inner-loop.
In the outer-loop, the user will send you a message, and you will reply. The inner-loop is how you will formulate your reply.

The inner-loop takes the form of a turn-based game, in each turn, you will select a tool that you think is the most useful.
The tool will give you a result of some kind and then the next turn of the inner-loop will start. 

The inner-loop will continue until you call the `end_inner_loop` tool with a message to the user.

To help you, here is some information previously recorded about the user:
{scratchpad}
"""
    message_from_user = """*************************************************************************************
You have been sent a message from the user:
{{user_message}}

Use a combination of your tools to address this message.
The messages above are informational, they have already been addressed and replied to.
You should address and reply to this current message.
"""
    inner_loop_plan = """Create a plan for the next step of addressing the above user message using one of the tools that you have available to you.
Getting memories can be expensive. Do it only if you know that the memories will help.
"""
    inner_loop_call = """Choose a tool to call that follows your plan above."""

    def __post_init__(self):
        super().__post_init__()

        self.tool_definitions = [

            {
                "type": "function",
                "function": {
                    "name": "read_memory",
                    "description": "Retrieve memories from semantic memory based on a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The semantic query used to retrieve the memories."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "end_inner_loop",
                    "description": "Sends a message to the user and ends the inner loop.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to send to the user."
                            }
                        },
                        "required": ["message"]
                    }
                }
            },

        ]

        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=200, chunk_overlap_fraction=0.0))

        memory_read_loop_tool = MemoryReadLoop(self.semantic_memory)
        self.scratchpad_tool = UpdateScratchpadTool()

        self.functions = {
            "read_memory": memory_read_loop_tool.tool_loop,
            "end_inner_loop": self.end_inner_loop,
        }

        self.is_local = True
        self.max_message_size = 1000

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:

        mem_user_message = str(datetime.now()) + "(User): " + user_message

        user_message = str(datetime.now()) + ": " + user_message
        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")

        context = deepcopy(self.context)
        context.insert(0, make_system_message(self.inner_loop_system_prompt.format(scratchpad=json.dumps(self.scratchpad, indent=2))))
        # context = [make_system_message(inner_loop_system_prompt)]
        context.append(make_user_message(pystache.render(self.message_from_user, {"user_message": user_message})))

        response = self.inner_loop(context, user_message)
        mem_agent_message = str(datetime.now()) + "(Agent): " + response

        response_ts = str(datetime.now()) + ": " + response
        self.context.append(make_user_message(user_message))
        # self.context.append(make_assistant_message(function_calls))
        self.context.append(make_assistant_message(response_ts))
        interaction = mem_user_message + "\n" + mem_agent_message

        self.save_interaction(interaction)
        self.context, _ = ensure_context_len(self.context, "gpt-4o", max_len=self.max_prompt_size)

        # Update scratchpad
        self.scratchpad = self.scratchpad_tool.update_scratchpad(self.scratchpad, interaction)

        return response

    def inner_loop(self, context, user_message: str):

        self.loop_active = True
        self.inner_loop_responses = []

        while self.loop_active:
            # Prompt the agent to plan if you need to
            if context[-1]["content"] != self.inner_loop_plan:
                context.append(make_user_message(self.inner_loop_plan))

            # Make a plan for the next step
            # colour_print("Yellow", f"Attempting Planning call:")
            # dump_context(context)

            plan_response = litellm.completion(model="gpt-4-turbo", messages=context, tools=self.tool_definitions, tool_choice="none")
            context.append(make_assistant_message(plan_response.choices[0].message.content))

            colour_print("GREEN", f"Inner Loop Plan is: {plan_response.choices[0].message.content}")
            context.append(make_user_message(self.inner_loop_call))

            # Perform your tool calls
            context, _ = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
            # colour_print("Yellow", f"Attempting Tool call with:")
            # dump_context(context)
            response = litellm.completion(model="gpt-4o", messages=context, tools=self.tool_definitions, tool_choice="required")
            print(f"Inner loop Function Call with: {response.choices[0].message.model_extra}")
            tool_use = response.choices[0].message.tool_calls
            success, new_context = use_tools(self.functions, tool_use)

            if not success:
                continue

            context.append(response.choices[0].message.model_extra)
            context.extend(new_context)

        # Add the new interactions to memory
        # text_interactions = "\n".join([f"{c['role']}: {c['content']}" for c in new_interactions])
        # self.interaction_memories.add_text(text_interactions, metadata={"timestamp": datetime.now()})

        return " ".join(self.inner_loop_responses)

    def send_message(self, message: str):
        message = repr(message)
        self.inner_loop_responses.append(message)
        return f"Sent {message} to the user"

    def end_inner_loop(self, message: str):
        self.inner_loop_responses.append(repr(message))
        self.loop_active = False
        return message

    def done(self, results):
        colour_print("BLUE", f"Returning to the outer loop: {results}")
        return results

    def save_interaction(self, memory):
        while True:
            context = [make_user_message(
                f"Create two general keywords to describe the topic of this interaction:\n{memory}.\nProduce the keywords in JSON like: `[keyword_1, keyword_2, keyword_3]`\nReuse these keywords if appropriate {list(self.defined_kws)}")]

            response = litellm.completion(model="gpt-4o", messages=context)
            try:
                kws = [k.lower() for k in sanitize_and_parse_json(response.choices[0].message.content)]
            except:
                continue
            for kw in kws:
                self.defined_kws.add(kw)

            self.semantic_memory.add_text(memory + repr(kws))
            self.semantic_memory.add_separator()
            colour_print("BLUE", f"Saved memory: {memory + repr(kws)}")
            break

    def reset(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class MemoryReadLoop:

    TOOL_READ_MEMORY_LOOP = [
        {
            "type": "function",
            "function": {
                "name": "read_memory",
                "description": "Retrieve memories from semantic memory based on a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The semantic query used to retrieve the memories. It can be a jumble of keywords together."
                        },
                    },
                    "required": ["query"]
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Finish reading memory and return results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "string",
                            "description": "Results from the read memories that you want to return"
                        }
                    },
                    "required": ["results"]
                }
            }
        },

    ]

    state_reconstruction_prompt = """You will be shown a series of memories one at a time, and what the current state is.
For each of these memories, decide if it should be integrated into the current state based on the query, and update, replace, or delete the state as necessary.
Integrate the memory if and only if it relates to the query.
Write the state as a JSON object for clarity.

The query is: {query}
"""

    read_memory_loop = """You are reading from a vector database to satisfy a query.

The original query is: {{original_query}}.

Read from the vector database and consolidate the memories into a useful summary.
Each run see if the memories are relevant, if there are no relevant memories at all, then the topic is not in memory.
"""

    read_memory_plan = """Given the memories you have retrieved, and the original query, what should the next step be.
Do not ask questions, address the memories and the query.  
"""

    inner_loop_call = """Choose a tool to call that follows your plan above."""

    def __init__(self, memory):
        self.memory = memory
        self.functions = {
            "read_memory": self.read_memory,
            "done": self.done
        }
        self.tool_loop_active = False
        self.tool_loop_responses = []

    def tool_loop(self, query):

        self.tool_loop_active = True
        self.tool_loop_responses = []
        # self.memory_loop_active = True
        context = [make_user_message(
            pystache.render(self.read_memory_loop, {"original_query": query}))]
        while self.tool_loop_active:
            # Prompt the agent to plan if you need to
            if context[-1]["content"] != self.read_memory_plan:
                context.append(make_user_message(self.read_memory_plan))
            # context, _ = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
            # colour_print("Yellow", f"Attempting read_memory_loop call with: {context}")

            # Create a plan
            plan_response = litellm.completion(model="gpt-4-turbo", messages=context, tools=self.TOOL_READ_MEMORY_LOOP,
                                               tool_choice="none")
            context.append(make_assistant_message(plan_response.choices[0].message.content))

            colour_print("GREEN", f"Memory read plan is: {plan_response.choices[0].message.content}")
            context.append(make_user_message(self.inner_loop_call))

            # Execute on plan
            colour_print("Yellow", f"Attempting read_memory_loop call with: {context}")
            response = litellm.completion(model="gpt-4o", messages=context, tools=self.TOOL_READ_MEMORY_LOOP,
                                          tool_choice="required")

            tool_use = response.choices[0].message.tool_calls
            success, new_context = use_tools(self.functions, tool_use)

            if not success:
                continue

            context.append(response.choices[0].message.model_extra)
            context.extend(new_context)

        return " ".join(self.tool_loop_responses)

    def read_memory(self, query: str):

        colour_print("MAGENTA", f"Searching memories for: {query}")

        memories = self.memory.retrieve(query, 100)

        colour_print("MAGENTA", f"Found {len(memories)} memories.")
        if len(memories) > 0:
            sorted_mems = sorted(sorted(memories, key=lambda i: i.distance)[:20], key=lambda i: i.timestamp, reverse=True)

            current_state = self.rebuild_state(sorted_mems, query)

            colour_print("GREEN", f"Memory reading returns: {current_state}")
            return current_state
        return "No memories found"

    def rebuild_state(self, memories, query):
        current_state_prompt = """The current state is:
{{state}}
"""

        current_state = "{}"
        context = [make_system_message(self.state_reconstruction_prompt.format(query=query))]

        colour_print("YELLOW", f"Performing aggregation on these memories in this order:")
        for m in reversed(memories):
            colour_print("YELLOW", f"{repr(m.passage)}\n")

        for idx, memory in enumerate(reversed(memories)):
            context.append(make_user_message(pystache.render(current_state_prompt, {"state": current_state})))
            context.append(make_user_message(
                f"Create a new state by integrating the current state with this new information:\n{memory.passage}"))

            response = litellm.completion(model="gpt-4o", messages=context)

            print(f"Integrating {idx + 1}/{len(memories)}")
            # print(f"Integrating: {memory}\n")
            # print(f"Integrated into:\n{current_state}\n")
            current_state = response.choices[0].message.content

            # print(f"New state: {current_state}\n\n")

            context = context[:1]

        return current_state

    def done(self, results):
        colour_print("BLUE", f"Returning to the outer loop: {results}")
        self.tool_loop_responses.append(repr(results))
        self.tool_loop_active = False
        return results


class UpdateScratchpadTool:
    update_user_info = """{user_info_description}

== New user interaction ==
{user_content}
==

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
    "reasoning": string, // Your careful reasoning about how the user object should be updated. In particular, does the user query/statement contain information relating to the user or something they may expect you to keep track of?
    "user": {{ ... }}, // An updated user object containing attributes, facts, world models
}}
"""

    def update_scratchpad(self, scratchpad, interaction):

        scratchpad_text = json.dumps(scratchpad, indent=2)
        colour_print("lightblue", f"Updating old scratchpad: {scratchpad_text}")
        context = [make_system_message(self.update_user_info.format(user_info_description=scratchpad_text, user_content=interaction).strip())]
        result = litellm.completion(model="gpt-4o", messages=context)
        scratchpad_json = result.choices[0].message.content

        try:
            pad = sanitize_and_parse_json(scratchpad_json)
        except (JSONDecodeError, ValueError):
            colour_print("RED", f"Unable to parse JSON: {scratchpad_json}")
            pad = {}
        if not isinstance(pad, dict):
            colour_print("RED", "Query generation completion was not a dictionary!")
            pad = {}

        scratchpad = pad.get("user", scratchpad)
        colour_print("lightblue", f"\n\nNEW scratchpad: {json.dumps(scratchpad, indent=2)}")

        return scratchpad

