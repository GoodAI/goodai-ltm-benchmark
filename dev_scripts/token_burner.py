import shutil
import requests
from time import time, sleep
from json import JSONDecodeError
from typing import Optional
from datetime import datetime
from googlesearch import search
from multiprocessing import Process, SimpleQueue
from utils.llm import ask_llm as _ask_llm
from utils.llm import LLMContext, LLMMessage, make_system_message, make_user_message, make_assistant_message
from utils.ui import multiline_input, colour_print
from utils.constants import DATA_DIR
from goodai.ltm.mem.auto import AutoTextMemory, BaseTextMemory
from goodai.helpers.json_helper import sanitize_and_parse_json

MODEL = "gpt-3.5-turbo"
INPUT_PREFIX = "> "
TMP_DATA_PATH = DATA_DIR.joinpath("token_burner")
AGENT_OUTPUT_FILE = TMP_DATA_PATH.joinpath("agent_output.txt")
LOG_FILE = TMP_DATA_PATH.joinpath("log.txt")  # Logs full agent's experience
TARGET_LOOP_SECONDS = 15
ALLOWED_ACTIONS = {"answer", "think", "search", "read", "nothing"}

system_prompt = """
You are an advanced AI agent.
Your goal is to help the user in whatever they need.
You don't have to answer questions right away. You can take your time to think and do some research first.
All events are timestamped (YYYY-MM-DD hh:mm:ss). Current time is {current_timestamp}.
You are aware of the passing of time and you can think about future instances of yourself and events that take place in the future.
At every step, you are expected to choose to perform one of the following actions:
- answer(message: str)  # Send a message back to the user
- think(topic: str, thoughts: str)  # Write down some thoughts. This should help you tackle hard tasks.
- search(search_string: str, num_results: int = 10)  # Search the Internet about anything.
- read(url: str)  # Access an URL and read its content
- nothing()  # Do nothing
Provide your answer in JSON form, like this:
{{
  "action": "action_name",
  "kwargs": {{
    "arg_name": <arg_value>,
    ...
  }}
}}
You don't need to timestamp your responses. Your responses will be timestamped automatically.
""".strip()


# The functions are actually implemented.
# After a couple of security checks, I call them via `vars()[f"agent_fn_{name}"](**kwargs)`.
def agent_fn_answer(message: str):
    send_message_to_user(message)


def agent_fn_think(topic: str, thoughts: str):
    pass


def agent_fn_search(search_string: str, num_results: int = 10) -> str:
    return "\n".join(search(search_string, num_results=num_results))


def agent_fn_read(url: str) -> str:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return f"Failed to retrieve content from {url}. Status code: {response.status_code}."
    except requests.RequestException as e:
        return f"An error occurred: {e}"


def agent_fn_nothing():
    pass


def str_timestamp(ts: Optional[datetime] = None) -> str:
    if ts is None:
        ts = datetime.now()
    return str(ts)[:-7]


def dynamic_system_prompt() -> LLMMessage:
    return make_system_message(system_prompt.format(current_timestamp=str_timestamp()))


def log(s: str, end="\n"):
    with LOG_FILE.open("a") as fd:
        fd.write(s)
        fd.write(end)


def update_log(context: LLMContext):
    logged_messages = 0
    if LOG_FILE.exists():
        with open(LOG_FILE) as fd:
            for line in fd:
                if line.startswith("ROLE: "):
                    logged_messages += 1
    with open(LOG_FILE, "a") as fd:
        for msg in context[1 + logged_messages:]:
            fd.write(f"ROLE: {msg['role']}\n")
            fd.write(msg["content"])
            fd.write("\n")


def ask_llm(context: LLMContext, **kwargs) -> str:
    log("LLM call")
    context[0] = dynamic_system_prompt()
    response = _ask_llm(context, temperature=0, **kwargs)
    log("LLM response received")
    context.append(make_assistant_message(f"{str_timestamp()}\n{response}"))
    update_log(context)
    return response


def add_user_message(message: str, context: LLMContext):
    context.append(make_user_message(f"{str_timestamp()}\n{message}"))
    update_log(context)


def add_error(error_str: str, context: LLMContext):
    context.append(make_system_message(f"{str_timestamp()}\nERROR: {error_str}"))
    update_log(context)


def send_message_to_user(message: str):
    with open(AGENT_OUTPUT_FILE, "a") as fd:
        fd.write(message)
        fd.write("\n")


def agent_step(context: LLMContext, ltm: BaseTextMemory):
    response = ask_llm(context)

    try:
        action = sanitize_and_parse_json(response)
    except (ValueError, JSONDecodeError):
        raise ValueError("Your response is not in JSON form.")

    if not isinstance(action, dict) or "action" not in action:
        raise ValueError("Your response is not in the right format.")

    if action["action"] not in ALLOWED_ACTIONS:
        raise ValueError(f"Unknown action {action['action']}")

    if action["action"] != "nothing" and "kwargs" not in action:
        raise ValueError("Missing function parameters.")

    log(f"Calling {action['action']}({', '.join(f'{k}={repr(v)}' for k, v in action.get('kwargs', {}).items())})")
    result = globals()[f"agent_fn_{action['action']}"](**action.get("kwargs", {}))

    if isinstance(result, str):
        context.append(make_system_message(f"{str_timestamp()}\n{result}"))
        update_log(context)


def agent_loop(in_queue: SimpleQueue, context: LLMContext, ltm: BaseTextMemory):
    add_user_message(in_queue.get(), context)  # The first user message triggers the start of the loop.
    while True:
        loop_t0 = time()
        while not in_queue.empty():
            add_user_message(in_queue.get(), context)
        try:
            agent_step(context, ltm)
        except Exception as exc:
            add_error(repr(exc), context)
            continue
        actual_duration = time() - loop_t0
        sleep_time = max(TARGET_LOOP_SECONDS - actual_duration, 0)
        sleep(sleep_time)


def main():
    context = [dynamic_system_prompt()]
    messages_queue = SimpleQueue()

    # Reset data path
    # shutil.rmtree(TMP_DATA_PATH, ignore_errors=True)
    # TMP_DATA_PATH.mkdir(parents=True)

    # The agent writes to a file, so it doesn't get in the way of the user's messages.
    # This file is displayed separately: tail -f agent_output.txt
    log("<agent is starting up>")

    # The agent runs in a separate thread, running at approx. 4 loops per second.
    agent_process = Process(
        target=agent_loop,
        kwargs=dict(
            in_queue=messages_queue,
            context=context,
            ltm=AutoTextMemory.create(),
        ),
    )
    agent_process.daemon = True
    agent_process.start()

    # The user can send messages at any time.
    # We capture these messages and put them into the agent's queue.
    colour_print("yellow", "Two empty lines signal the end of the message.\nEnter 'exit' to terminate the run.")
    while True:
        colour_print("green", "User: ", end="")
        message = multiline_input("", in_prefix=INPUT_PREFIX)
        if message.strip().lower() == "exit":
            break
        messages_queue.put(message)


if __name__ == "__main__":
    main()
