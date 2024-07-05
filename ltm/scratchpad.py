import json
import time
from typing import Callable, Any, Iterator
from copy import deepcopy
from utils.llm import make_user_message, LLMContext
from goodai.helpers.json_helper import sanitize_and_parse_json


query_rewrite_template = """
{user_info_description}

== New user question/statement ==
{user_content}
==

Based on prior user information and the above user question/statement, your task is to provide semantic queries for searching archived conversation history that may be relevant to reply to the user.
The search queries you produce should be compact reformulations of the user question/statement, taking context into account.
The purpose of the queries is accurate information retrieval.
Search is purely semantic.
Time-based queries are unsupported.

Write JSON in the following format:
{{
    "queries": array // An array of strings: 1 or 2 descriptive search phrases
}}
""".strip()

system_prompt_template = """
You are an expert in helping AI assistants manage their knowledge about a user and their operating environment.
These AI assistants keep a JSON structure with user information that they use as a scratchpad.

Now an AI assistant requires your expertise.
{last_messages}

== New user question/statement ==
{message}
==

This AI assistant has just received the above user question/statement. Based on it, your task is to keep the JSON scratchpad up to date.
The scratchpad is expected to contain information provided by the user, such as facts about themselves or information they are expecting the AI assistant to keep track of. The scratchpad is specially well suited for quickly-changing or temporal information.
Avoid storing unimportant general knowledge that any AI assistant should be already aware of.
Capture information provided by the user without omitting important details.
Property names should be descriptive.

You will address this task in turns. You will be guided through the process.
""".strip()

changes_yesno_template = """
This is what the user information looks like now:
{user_info}

Would you like to add new information or update any part of it? Answer "yes" or "no".
""".strip()

single_change_template = """
Let's update the user information then, one item at a time.
You can either add new content or update an existing item. If updating, the full item's content will be replaced by "new_content".
Only apply small and concise changes and avoid duplicates. Summarize large items if you really want to keep the info, otherwise empty them or set them to null.
Provide JSON text indicating the content to store:
{
  "key": "key_to_update",
  "new_content": ... // Any JSON object
}
If the item is nested, provide the full key with slashes, like a path: "A/B/C"
""".strip()

item_use_question_template = """
Now, let's focus on your last response: {response}
What motivated you to give it exactly that way?
More specifically, what information from the user has been useful or relevant for it?
Provide a JSON list of 5 strings max, pointing to items of the JSON structure that you keep in your system prompt.
If the item is nested, provide the full key with slashes, like a path: "A/B/C"
""".strip()


def to_text(scratchpad: dict) -> str:
    return json.dumps(scratchpad, indent=2)


def extract_json_dict(response: str) -> dict:
    i = response.find("{")
    j = response.rfind("}")
    return sanitize_and_parse_json(response[i:j+1])


def all_key_paths(node: Any) -> Iterator[str]:
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        yield key
        for key_path in all_key_paths(value):
            yield f"{key}/{key_path}"


def update_used_timestamp(scratchpad_timestamps: dict[str, float], scratchpad: dict, key_path: str | list[str], timestamp: float = None):
    timestamp = timestamp or time.time()
    keys = key_path if isinstance(key_path, list) else key_path.split("/")

    # Reach pointed node while registering partial routes
    parent_node = scratchpad
    for i, ki in enumerate(keys):
        if not isinstance(parent_node, dict) or ki not in parent_node:
            return
        parent_node = parent_node[ki]
        k = "/".join(keys[:i + 1])
        scratchpad_timestamps[k] = timestamp

    # Include all levels below too
    if isinstance(parent_node, dict):
        root_path = "/".join(keys)
        for key_path in all_key_paths(parent_node):
            scratchpad_timestamps[f"{root_path}/{key_path}"] = timestamp


def add_new_content(scratchpad_timestamps: dict[str, float], scratchpad: dict, key_path: str, content: Any):

    # Navigate to the insertion point and add content
    keys = key_path.split("/")
    parent_node = scratchpad
    for k in keys[:-1]:
        if k not in parent_node:
            parent_node[k] = dict()
        parent_node = parent_node[k]
    parent_node[keys[-1]] = content

    # Register this change -> update items' timestamps
    update_used_timestamp(scratchpad_timestamps, scratchpad, keys)


def remove_item(scratchpad: dict, key_path: str):
    keys = key_path.split("/")
    parent_node = scratchpad
    for k in keys[:-1]:
        if k not in parent_node:
            return
        parent_node = parent_node[k]
        if not isinstance(parent_node, dict):
            return
    parent_node.pop(keys[-1], None)


def register_used_items(agent: "LTMAgent", context: LLMContext, cost_cb: Callable[[float], None]):
    """Not used. We discovered that this hurt performance."""
    ask_kwargs = dict(temperature=0.01, label="scratchpad", cost_callback=cost_cb)
    context = deepcopy(context)
    context.append(make_user_message(item_use_question_template.format(response=context[-1]["content"])))
    response = agent._truncated_completion(context, **ask_kwargs)
    try:
        key_list = sanitize_and_parse_json(response)
    except:
        return
    if not isinstance(key_list, list):
        return
    if len(key_list) > 5:
        print("--- MORE THAN 5 KEYS GIVEN!!! ---")
        print(f"{len(key_list)} keys given")
    now = time.time()
    for key_path in key_list:
        update_used_timestamp(agent.user_info_ts, agent.user_info, key_path, timestamp=now)
