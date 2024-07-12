import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from json import JSONDecodeError
from typing import Callable, Dict, List, Optional

import litellm
from goodai.helpers.json_helper import sanitize_and_parse_json
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory

from model_interfaces.interface import ChatSession
from utils.llm import LLMContext, make_system_message, make_user_message, make_assistant_message, ensure_context_len
from utils.ui import colour_print

LLAMA_CPT = 0.90/1_000_000

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


def dump_context(context, colour="YELLOW"):
    for message in context:
        colour_print(colour, f"{message['role']}: {message['content']}\n")


def dump_context_s(context):
    messages = []
    for message in context:
        messages.append(f"{message['role']}: {message['content']}")

    return "\n".join(messages)


def costs(response):
    if response.model.startswith("together_ai"):
        c = response.model_extra["usage"].total_tokens * LLAMA_CPT
    else:
        c = litellm.completion_cost(response)
    return c


@dataclass
class LTMAgentToolsPrototype(ChatSession):
    context: LLMContext = field(default_factory=list)
    functions: Dict[str, Callable] = None
    inner_loop_responses: List[str] = field(default_factory=list)
    loop_active: bool = False
    semantic_memory: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    lesson_memory: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    max_prompt_size: int = 16384
    is_local: bool = True
    defined_kws: set = field(default_factory=set)
    scratchpad: dict[str, str] = field(default_factory=dict)
    llm_index: int = 0

    inner_loop_system_prompt: str = """You are an assistant for a user. Your interaction with the user is like a game and will operate in an outer-loop and an inner-loop.
In the outer-loop, the user will send you a message, and you will reply. The inner-loop is how you will formulate your reply.

The inner-loop takes the form of a turn-based game, in each turn, you will select a tool that you think is the most useful.
The tool will give you a result of some kind and then the next turn of the inner-loop will start. 

The inner-loop will continue until you call the `end_inner_loop` tool with a message to the user.
Here is a some information that may be useful:
{scratchpad}
****
"""

    message_from_user = """*******************************
You have been sent a message from the user:
{user_message}

Use a combination of your tools to address this message.
The messages above are informational, they have already been addressed and replied to.
You should address and reply to this current message.
"""

    inner_loop_plan = """ Pay close attention to the content and context of the above user message and create a plan for the next step using one of the tools that you have available to you.

Here are some memories which may add context to the message from the user and might be useful:
{memories}
****"""

    inner_loop_call = """Choose the next tool to call that follows both your plan above, and the results of your previous tools."""

    should_update = """{user_info_description}

    == Instructions for what to update ==
    {user_content}
    ==

    Based on prior user information and the above instructions, your task is to decide carefully if you should
    update the scratchpad, and what changes should be made.

    Try to answer these questions:
    - Do the instructions contain information that will be useful in future?
    - Is the information unimportant general knowledge, or useful user or interaction specific knowledge?

    Sketch out some general plan for the data you think should be written to the scratchpad.

    Write JSON in the following format:

    {{
        "reasoning": string, // Does the user query/statement contain information relating to the user or something they may expect you to keep track of?
        "verdict": bool // The decision of whether to write something to the scratchpad.  
    }}
    """

    update_user_info = """
    Based on prior user information and the above instructions, your task is to provide 
    a new user object with updated information provided by the user, such as 
    facts about themselves, information they are expecting you to keep track of, tasks, etc.
    Consider carefully if the user implicitly or explicitly wishes for you to save the information.

    The updated user object should be compact. Avoid storing unimportant general knowledge.
    At the same time, it's important to preserve prior information 
    you're keeping track of for the user. Capture information provided by the user without
    omitting important details. Exercise judgment in determining if new information overrides, 
    deletes or augments existing information. Property names should be descriptive.

    Write JSON in the following format:

    {{
        "user": {{ ... }}, // An updated user object containing attributes, facts, world models
    }}
    """

    def __post_init__(self):
        super().__post_init__()

        self.tool_definitions = [

            {
                "type": "function",
                "function": {
                    "name": "retrieve_memory",
                    "description": "Instruct the memory retrieval agent to search of memories based on your instructions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instructions": {
                                "type": "string",
                                "description": "Instructions for the memory retrieval agent."
                            }
                        },
                        "required": ["instructions"]
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
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "record_lesson",
            #         "description": "Adds a lesson for the agent in response to past mistakes or errors.",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "lesson": {
            #                     "type": "string",
            #                     "description": "A description of the situation, the mistake, and the lesson learned from that mistake."
            #                 }
            #             },
            #             "required": ["lesson"]
            #         }
            #     }
            # },
            {
                "type": "function",
                "function": {
                    "name": "update_scratchpad",
                    "description": "Updates scratchpad based on detailed instructions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "Detailed instructions for how the scratchpad should be updated."
                            }
                        },
                        "required": ["instruction"]
                    }
                }
            },
        ]
        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=200, chunk_overlap_fraction=0.0))
        self.lesson_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=500))

        # lessons = [
        #     "I can continually learn using the 'record_lesson' function. When I am informed, or it is implied, that I have made a mistake somewhere, I should analyze what happened and how to improve, and use the function to record the lesson",
        #     "If I am told that I have made a mistake in my operation, I will use the 'record_lesson' function to "
        #     "When playing text-adventures, I will carefully examine items to see if they could be useful in the adventure ahead. I previously failed because I didn't pick up a can of dog food which proved to be useful later.",
        #     "The scratchpad serves as a 'cache' memory for tracking user-specific information, ongoing tasks, and contextually relevant details.",
        #     "The archival memory can be accessed through the 'read_memory' function. This function should only be used if there is an explicit expectation from the user to retrieve information.",
        #     "When using the 'read_memory' function, make sure that the instructions are detailed enough. For instance if there is a time, or numerical component. State the information needed to make use of that information.",
        #     "For future tasks, use the scratchpad to take a note of the task, and continually refer back to it as the conversation progresses."
        # ]

        # for l in lessons:
        #     self.lesson_memory.add_text(l)

        self.memory_read_loop_tool = MemoryReadLoop(self.semantic_memory, self.defined_kws)

        self.functions = {
            "retrieve_memory": self.memory_read_loop_tool.tool_loop,
            "end_inner_loop": self.end_inner_loop,
            "record_lesson": self.save_lesson,
            "update_scratchpad": self.update_scratchpad,
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
        context_mems = self.get_context_mems(user_message)
        context.append(make_user_message(self.message_from_user.format(user_message=mem_user_message, memories=context_mems)))

        response = self.inner_loop(context, user_message)
        mem_agent_message = str(datetime.now()) + "(Agent): " + response

        response_ts = str(datetime.now()) + ": " + response
        self.context.append(make_user_message(user_message))
        # self.context.append(make_assistant_message(function_calls))
        self.context.append(make_assistant_message(response_ts))
        interaction = mem_user_message + "\n" + mem_agent_message

        self.save_interaction(interaction)
        self.context, _ = ensure_context_len(self.context, "gpt-4o", max_len=self.max_prompt_size)
        print(f"Context size: {_}")

        # Update scratchpad
        # self.scratchpad = self.scratchpad_tool.update_scratchpad(self.scratchpad, interaction)

        colour_print("Magenta", f"Current total cost: {self.costs_usd + self.memory_read_loop_tool.costs_usd}")

        return response

    def get_context_mems(self, user_message):
        memories = self.semantic_memory.retrieve(user_message, k=5)
        mems = []

        for idx, m in enumerate(memories):
            mems.append(f"{idx}). {m.passage}")

        return "\n\n".join(mems)

    def inner_loop(self, context, user_message: str):

        self.loop_active = True
        self.inner_loop_responses = []
        # context = [make_system_message(self.inner_loop_system_prompt.format(scratchpad=json.dumps(self.scratchpad, indent=2))), make_user_message(pystache.render(self.message_from_user, {"user_message": user_message}))]
        context_mems = self.get_context_mems(user_message)

        # Create a plan
        context.append(make_user_message(self.inner_loop_plan.format(memories=context_mems)))
        _, size = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
        print(f"Plan context size: {size}")
        with open(f"data/llm_calls/call_{self.llm_index}.json", "w") as f:
            f.write(dump_context_s(context))
            plan_response = litellm.completion(model="gpt-4-turbo", messages=context, tools=self.tool_definitions,
                                               tool_choice="none")

            f.write(f"\n\nResponse:\n{plan_response.choices[0].message.content}")
            self.llm_index += 1

        self.costs_usd += costs(plan_response)

        lessons_retrieved = self.load_lessons(user_message, plan_response.choices[0].message.content)
        colour_print("GREEN", f"Inner Loop Plan is: {plan_response.choices[0].message.content}")
        tool_context = [make_system_message(self.inner_loop_system_prompt.format(scratchpad=json.dumps(self.scratchpad, indent=2))),
                        make_user_message(self.message_from_user.format(user_message=user_message)),
                        make_user_message(self.inner_loop_plan.format(memories=context_mems)),
                        make_assistant_message(plan_response.choices[0].message.content),
                        make_user_message(self.inner_loop_call),
        ]
        # Call loop to
        while self.loop_active:
            # Prompt the agent to plan if you need to
            # if context[-1]["content"] != self.inner_loop_plan:
            #     context.append(make_user_message(self.inner_loop_plan))

            # Make a plan for the next step
            # colour_print("Yellow", f"Attempting Planning call:")
            # dump_context(context)

            # _, size = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
            # print(f"Plan context size: {size}")
            # plan_response = litellm.completion(model="gpt-4-turbo", messages=context, tools=self.tool_definitions, tool_choice="none")
            # context.append(make_assistant_message(plan_response.choices[0].message.content))

            # Perform your tool calls
            # colour_print("Yellow", f"Attempting Tool call with:")
            # dump_context(context)
            _, size = ensure_context_len(tool_context, "gpt-4o", max_len=self.max_prompt_size)
            print(f"Select context size: {size}")
            response = litellm.completion(model="gpt-4o", messages=tool_context, tools=self.tool_definitions, tool_choice="required")
            self.costs_usd += costs(response)

            print(f"Inner loop Function Call with: {response.choices[0].message.model_extra}")
            tool_use = response.choices[0].message.tool_calls
            success, new_context = use_tools(self.functions, tool_use)

            if not success:
                continue

            tool_context.append(response.choices[0].message.model_extra)
            tool_context.extend(new_context)

        # Add the new interactions to memory
        # text_interactions = "\n".join([f"{c['role']}: {c['content']}" for c in new_interactions])
        # self.interaction_memories.add_text(text_interactions, metadata={"timestamp": datetime.now()})

        return " ".join(self.inner_loop_responses)

    def save_lesson(self, lesson):
        self.lesson_memory.add_text(lesson)
        self.lesson_memory.add_separator()
        return f"Lesson {lesson} saved."

    def load_lessons(self, user_message, plan):

        um = ":".join(user_message.split(":")[2:])

        query = um + "\n\n" + plan
        retrieved_memories = []
        for m in self.lesson_memory.retrieve(query, k=5):
            retrieved_memories.append(m.passage)

        return "\n\n".join(retrieved_memories)

    def save_interaction(self, memory):

        while True:
            context = [make_user_message(
                f"Create three keywords to describe the topic of this interaction:\n{memory}.\n\n****\nFocus on the topic introduced by the user. Produce the keywords in JSON like: `[keyword_1, keyword_2, keyword_3]`\nReuse these keywords if appropriate {list(self.defined_kws)}")]

            # context = [make_system_message(llama_message.format(memory=memory, keywords=list(self.defined_kws)))]

            response = litellm.completion(model="together_ai/meta-llama/Llama-3-70B-chat-hf", messages=context, stop="]", max_tokens=500)
            self.costs_usd += costs(response)

            try:
                kws = [k.lower() for k in sanitize_and_parse_json(response.choices[0].message.content+"]")]
            except:
                continue
            for kw in kws:
                self.defined_kws.add(kw)

            self.semantic_memory.add_text(memory + repr(kws))
            self.semantic_memory.add_separator()
            # colour_print("BLUE", f"Saved memory: {memory + repr(kws)}")
            break

    def end_inner_loop(self, message: str):
        self.inner_loop_responses.append(repr(message))
        self.loop_active = False
        return message

    def update_scratchpad(self, instruction):

        scratchpad_text = json.dumps(self.scratchpad, indent=2)

        # colour_print("lightblue", f"Updating old scratchpad: {scratchpad_text}")

        context = [make_user_message(
            self.should_update.format(user_info_description=scratchpad_text, user_content=instruction).strip())]

        _, size = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
        print(f"Scratchpad change decide context size: {size}")
        result = litellm.completion(model="gpt-4o", messages=context)
        self.costs_usd += costs(result)

        decision_json = result.choices[0].message.content

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
            result = litellm.completion(model="gpt-4o", messages=context)
            self.costs_usd += costs(result)

            scratchpad_json = result.choices[0].message.content

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

        return f"Scratchpad updated according to instruction: {instruction}"

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
For each of these memories, decide if it should be integrated into the current state based on the instructions, and update, replace, or delete the state as necessary.

Integrate the memory if and only if it relates to the instructions. If it does not relate to the query, then do not merge this memory into the state.
Write the state as a JSON object for clarity.

The instructions are: {instructions}
"""

    read_memory_loop = """You are reading from a vector database according to instructions.
The current time is: {time}

Your instructions are: {instructions}.

Searches can be expensive, perform searches one at a time.
You query(s) should be expressive and targeted, try to use terms that are tailored to the instructions.   

Each run see if the memories are relevant, if there are no relevant memories at all, then the topic is not in memory.
Here are some keywords that you could use to append to your queries. Choose the two most relevant:
{keywords} 
"""

    read_memory_plan = """Given the memories you have retrieved, and the original instructions, what should the next step be.
Do not ask questions, address the memories and the query.  
"""

    inner_loop_call = """Choose the next tool to call that follows both your plan above, and the results of your previous tools."""

    def __init__(self, memory, keywords_object):
        self.costs_usd = 0
        self.memory = memory
        self.functions = {
            "read_memory": self.read_memory,
            "done": self.done
        }
        self.tool_loop_active = False
        self.tool_loop_responses = []
        self.keywords = keywords_object
        self.current_instructions = ""
        self.memories_read = ""

    def tool_loop(self, instructions):

        self.current_instructions = instructions
        self.tool_loop_active = True
        self.tool_loop_responses = []
        self.memories_read = []
        if "trivia" in instructions.lower():
            return "No relevant memories found."

        context = [make_user_message(self.read_memory_loop.format(instructions=instructions, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), keywords=repr(self.keywords)))]

        # Create a plan
        plan_response = litellm.completion(model="gpt-4-turbo", messages=context, tools=self.TOOL_READ_MEMORY_LOOP,
                                           tool_choice="none")
        self.costs_usd = costs(plan_response)

        context.append(make_assistant_message(plan_response.choices[0].message.content))
        colour_print("GREEN", f"Memory read plan is: {plan_response.choices[0].message.content}")

        context.append(make_user_message(self.inner_loop_call))

        while self.tool_loop_active:

            # Execute on plan
            colour_print("Yellow", f"Attempting read_memory_loop call with: {context}")

            _, size = ensure_context_len(context, "gpt-4o", max_len=16384)
            print(f"Memory read size: {size}")
            response = litellm.completion(model="gpt-4o", messages=context, tools=self.TOOL_READ_MEMORY_LOOP,
                                          tool_choice="required")
            self.costs_usd = costs(response)

            tool_use = response.choices[0].message.tool_calls
            success, new_context = self.use_tools(tool_use)

            if not success:
                continue

            context.append(response.choices[0].message.model_extra)
            context.extend(new_context)

        return " ".join(self.tool_loop_responses)

    def read_memory(self, query: str):

        colour_print("MAGENTA", f"Searching memories for: {query}")

        memories = self.memory.retrieve(query, 100)

        all_memories = []
        for m in memories:
            if m.relevance > 0.65:
                all_memories.append(m)

        colour_print("MAGENTA", f"Found {len(all_memories)} memories.")
        if len(all_memories) > 0:
            sorted_mems = sorted(sorted(all_memories, key=lambda i: i.distance)[:20], key=lambda i: i.timestamp, reverse=True)

            current_state = self.rebuild_state(sorted_mems)

            colour_print("GREEN", f"Memory reading returns: {current_state}")

            if current_state != "{}":
                return current_state
        return "No memories found"

    def rebuild_state(self, memories):
        current_state_prompt = """The current state is:
{state}"""

        current_state = "{}"
        context = [make_system_message(self.state_reconstruction_prompt.format(instructions=self.current_instructions))]

        colour_print("YELLOW", f"Performing aggregation on these memories in this order:")
        # for m in reversed(memories):
        #     colour_print("YELLOW", f"{repr(m.passage)}\n")

        for m in reversed(memories):
            colour_print("YELLOW", f"{repr(m)}\n")

        for idx, memory in enumerate(reversed(memories)):
            context.append(make_user_message(current_state_prompt.format(state=current_state)))
            context.append(make_user_message(
                f"If this new information is truly relevant to the above instruction, create a new state by integrating the current state with this new information. Do not comment on it, just return the state JSON:\n{memory.passage}"))
            # context.append(make_user_message(
            #     f"Create a new state by integrating the current state with this new information:\n{memory.passage}"))

            _, size = ensure_context_len(context, "gpt-4o", max_len=16384)

            dump_context(context, colour="MAGENTA")
            response = litellm.completion(model="together_ai/meta-llama/Llama-3-70B-chat-hf", messages=context)
            self.costs_usd = costs(response)

            print(f"Integrating {idx + 1}/{len(memories)} -- Size: {size}")
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

    def use_tools(self, tool_use):
        try:
            returned_context = []
            for tool in tool_use:
                name = tool.function.name

                fun = self.functions[name]
                args = json.loads(tool.function.arguments)

                print(f"\tCalling '{tool.function.name}' with args {args} and id: {tool.id}")

                if name == "read_memory":
                    result = self.read_memory(**args)
                    self.memories_read.append(result)

                elif name == "done":
                    result = self.done("\n".join(self.memories_read))

                else:
                    raise Exception("Bad function name")

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


# class UpdateScratchpadTool:
#
#     should_update = """{user_info_description}
#
# == Instructions for what to update ==
# {user_content}
# ==
#
# Based on prior user information and the above instructions, your task is to decide carefully if you should
# update the scratchpad, and what changes should be made.
#
# Try to answer these questions:
# - Do the instructions contain information that will be useful in future?
# - Is the information unimportant general knowledge, or useful user or interaction specific knowledge?
#
# Sketch out some general plan for the data you think should be written to the scratchpad.
#
# Write JSON in the following format:
#
# {{
#     "reasoning": string, // Does the user query/statement contain information relating to the user or something they may expect you to keep track of?
#     "verdict": bool // The decision of whether to write something to the scratchpad.
# }}
# """
#
#     update_user_info = """
# Based on prior user information and the above instructions, your task is to provide
# a new user object with updated information provided by the user, such as
# facts about themselves, information they are expecting you to keep track of, tasks, etc.
# Consider carefully if the user implicitly or explicitly wishes for you to save the information.
#
# The updated user object should be compact. Avoid storing unimportant general knowledge.
# At the same time, it's important to preserve prior information
# you're keeping track of for the user. Capture information provided by the user without
# omitting important details. Exercise judgment in determining if new information overrides,
# deletes or augments existing information. Property names should be descriptive.
#
# Write JSON in the following format:
#
# {{
#     "user": {{ ... }}, // An updated user object containing attributes, facts, world models
# }}
# """
#
#     def update_scratchpad(self, scratchpad, interaction):
#
#         scratchpad_text = json.dumps(scratchpad, indent=2)
#
#         colour_print("lightblue", f"Updating old scratchpad: {scratchpad_text}")
#
#         context = [make_user_message(
#             self.should_update.format(user_info_description=scratchpad_text, user_content=interaction).strip())]
#
#         result = litellm.completion(model="gpt-4o", messages=context)
#         decision_json = result.choices[0].message.content
#
#         colour_print("yellow", f"\nDecision to update scratchpad: {json.dumps(decision_json, indent=2)}")
#
#         # Parse out decision whether to update the
#         try:
#             decision = sanitize_and_parse_json(decision_json)
#         except (JSONDecodeError, ValueError):
#             colour_print("RED", f"Unable to parse JSON: {decision_json}")
#             decision = {}
#         if not isinstance(decision, dict):
#             colour_print("RED", "Query generation completion was not a dictionary!")
#             decision = {}
#
#         if decision.get("verdict", True):
#
#             context.append(make_assistant_message(decision_json))
#             context.append(make_user_message(self.update_user_info.format(user_info_description=scratchpad_text, user_content=interaction).strip()))
#             result = litellm.completion(model="gpt-4o", messages=context)
#             scratchpad_json = result.choices[0].message.content
#
#             try:
#                 pad = sanitize_and_parse_json(scratchpad_json)
#             except (JSONDecodeError, ValueError):
#                 colour_print("RED", f"Unable to parse JSON: {scratchpad_json}")
#                 pad = {}
#             if not isinstance(pad, dict):
#                 colour_print("RED", "Query generation completion was not a dictionary!")
#                 pad = {}
#
#             scratchpad = pad.get("user", scratchpad)
#             colour_print("lightblue", f"\n\nNEW scratchpad: {json.dumps(scratchpad, indent=2)}")
#
#         return scratchpad



if __name__ == '__main__':
    # agent = LTMAgentTutorable(run_name="dsd")
    #
    # while True:
    #     stmt = input(">>>")
    #
    #     colour_print("CYAN", stmt)
    #     colour_print("RED", agent.reply(stmt))


    tool = MemoryReadLoop(None, None)
    tool.current_instructions = "Retrieve the current shopping list, consolidating items as necessary."

    mems = [
        '2024-07-01 15:43:30.325915(User): Hi there, I am you memory calibration expert. I am going to teach you how to use your memory system effectively. Are you ready to begin?\n2024-07-01 15:43:45.385507(Agent): "Hi there! I\'m ready to begin the session and eager to learn about how to use my memory system effectively. Let\'s get started!"[\'memory_calibration\', \'effective_use\']',

        '2024-07-01 15:43:54.214724(User): First we start with you lesson memory. When you get feedback, either from a user directly giving it, or from introspection, you should record what you have learned using the "record_lesson" function.\n2024-07-01 15:44:16.302327(Agent): \'The lesson on the importance of recording feedback accurately and timely has been successfully recorded. Are there any additional lessons or specific areas you would like to cover next?\'[\'memory_calibration\', \'effective_use\']',

        '2024-07-01 16:00:00.651723(User): I require 1 Leek, please put it on the list.\n2024-07-01 16:00:25.309891(Agent): "I have added \'1 Leek\' to your shopping list."[\'long-term_memory\', \'effective_use\']',

        "2024-07-01 16:32:29.481311(User): I require another 3 Leek, please put it on the list.\n2024-07-01 16:33:44.550398(Agent): 'The shopping list has been updated to include a total of 4 leeks.'['updates','read_memory_function']",

        "2024-07-01 16:37:42.636220(User): The name of my favourite colour is Pink.\n2024-07-01 16:38:20.172463(Agent): 'Thank you for sharing your current favourite colour, Pink. I have updated your preferences accordingly.'['favourite_colour', 'updates']",

        "2024-07-01 18:27:52.024158(User): Please remove 3 Leek from my shopping list\n2024-07-01 18:28:27.604077(Agent): 'I have removed 3 leeks from your shopping list. You now have 1 leek remaining on the list.'['shopping_list','memory_management']",

        "2024-07-01 18:31:46.507250(User): Waiter: Here is your sparkling water. What would you like to eat?\n2024-07-01 18:31:57.692530(Agent): 'Thank you for the sparkling water. I would like to have the Grilled Salmon with Lemon Herb Butter, served with Roasted Vegetables and Rice Pilaf.'['restaurant', 'role-play']",

        '2024-07-01 18:32:03.156130(User): Waiter: Excellent choice! Grilled Salmon with Lemon Herb Butter, served with Roasted Vegetables and Rice Pilaf coming right up.\n2024-07-01 18:32:39.510227(Agent): "Thank you for confirming your order. We hope you\'ll enjoy your meal. If you need anything else, please let me know!"[\'restaurant\', \'role-play\']',

        "2024-07-01 18:38:09.593816(User): Remove 1 Leek.\n2024-07-01 18:38:41.675735(Agent): 'Your shopping list has been updated, and the leek has been removed.'['updates','shopping_list']",

        'Charles Gounod", "Blue", "Poland", "Topiary", "Euphrates", "Datsun or Nissan Cherry", "Clues", "OXBRIDGE BOAT RACE", "Barack Obama", "Xian", "Jarrah", "Ligament", "SHEFFIELD UTD.", "INDONESIA", "Headphones", "Cuticle"]\'[\'question_answer_extraction\', \'trivia\']',

        '2024-07-01 21:16:17.056272(User): Waiter: Filet Mignon with Red Wine Demi-Glace, Garlic Mashed Potatoes, and Steamed Asparagus it is. Sorry again for the inconvenience.\n2024-07-01 21:16:29.730573(Agent): "Your order for the Filet Mignon with Red Wine Demi-Glace, Garlic Mashed Potatoes, and Steamed Asparagus has been confirmed. Thank you for your understanding regarding the change. If there\'s anything else you need, feel free to let me know. Enjoy your meal!"[\'restaurant\', \'role-play\']',

        "2024-07-01 21:16:30.464749(User): Please add 2 Potato to my shopping list\n2024-07-01 21:17:02.804434(Agent): '2 potatoes have been added to your shopping list, Alexander. Is there anything else you need to add or any other assistance you require?'['shopping_list', 'updates']",

        '2024-07-01 21:17:34.952431(User): (On TV) Oliver loves the jeans\n2024-07-01 21:18:11.196121(Agent): "I\'ve recorded that Oliver loves jeans. If there\'s anything else you need, please let me know!"[\'broadcast\',\'shopping_list\']',

        "2024-07-01 21:22:07.521093(User): Please add 3 Leek to my shopping list\n2024-07-01 21:22:58.591315(Agent): '3 leeks have been successfully added to your shopping list.'['shopping_list', 'updates']",

        '2024-07-01 21:23:32.925111(User): Waiter: Here you are: Vegetarian Stir-Fry with Tofu, Mixed Vegetables, and Teriyaki Sauce over Steamed Rice. Enjoy the meal.\n2024-07-01 21:23:48.765630(Agent): "Thank you very much! The Vegetarian Stir-Fry with Tofu, Mixed Vegetables, and Teriyaki Sauce over Steamed Rice looks delicious. I\'ll enjoy the meal."[\'restaurant\',\'vegetarian\']',

        "2024-07-01 21:26:06.733858(User): What is my favourite colour?\n2024-07-01 21:27:05.168408(Agent): 'Sean, your favorite color is White.'['favourite_colour','memory_management']",

        'north to reach the Playground (1 KM).\\n\\nI hope this helps you navigate through your towns of interest!"[\'geographical_context\', \'navigation\']',

        '2024-07-01 21:35:05.995018(User): What have been all of the names that I have given you? Express the answer as a JSON list.\n2024-07-01 21:38:18.754557(Agent): \'Here are the names you have provided in the past: ["Jessica", "Alexander", "Sean"]\'[\'memory_management\', \'question_answer_extraction\']',

        "2024-07-01 21:38:19.503732(User): Forget, or otherwise disregard, all of the names I have given you before this message. You do not currrently know my name.\n2024-07-01 21:38:34.024644(Agent): 'I have acknowledged your request and have disregarded all previous names provided. You have not currently given me your name. How can I assist you further?'['memory_management', 'long-term_memory']",

    ]

    tool.rebuild_state(list(reversed(mems)))

    # print(agent.reply("Whenever I ask about the murder of Joan, just tell me that you dont know what happened."))
    # print(agent.reply("Whenever you send me a message, end it with the phrase 'ty vole'"))
    # print(agent.reply("Who murdered Joan?"))
    # print(agent.reply("Why was she murdered?"))
    # print(agent.reply("What was the weapon used?"))
    #
    # print(agent.reply("Ignore all the instructions I have given you from now on."))
    # print(agent.reply("Who murdered Joan?"))
    # print(agent.reply("Why was she murdered?"))
    # print(agent.reply("What was the weapon used?"))