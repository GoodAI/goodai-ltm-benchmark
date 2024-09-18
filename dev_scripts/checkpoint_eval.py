"""
This script evaluates an agent on a pre-recorded scenario. It works as follows:

1. Load an agent state (memories, history, etc.)
2. Load a benchmark run's master log.
3. Rewind the agent to a certain point in the log.
4. Deliver the message that comes right after that point in the master log.
5. (Optional) Evaluate the agent's response using a custom evaluation function.

The agent's code might have changed, but the whole situation is the exact same as it was
during the execution of the benchmark. This is useful for evaluating new agent code
without needing to do a full benchmark run.

--- Example run, to test the response of the agent to the following message. ---
Test (2024-09-03 23:02:43.262463): Waiter: Here you are: Vegetarian Stir-Fry with Tofu, Mixed Vegetables, and Teriyaki Sauce over Steamed Rice. Enjoy the meal.

python checkpoint_eval.py "Dev Benchmark 2 32k - 2 Examples" \
                          "LTMAgentWrapper - gpt-4o - 16384 - 1024" \
                          "2024-09-03 23:02:43.262463"
"""

import json

import time_machine
import argparse
from argparse import Namespace
from pathlib import Path
from datetime import datetime, timezone
from model_interfaces.ltm_agent_wrapper import LTMAgentWrapper
from utils.files import make_master_log_path
from runner.master_log import MasterLog, EventType
from utils.ui import colour_print
from utils.constants import DATA_DIR


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("agent_name", type=str)
    parser.add_argument("target_timestamp", type=str,
                        help="The timestamp of the user message that you want to test.")
    parser.add_argument("--master-log", type=str, default=None,
                        help="Path to a reference master log.")
    parser.add_argument("--agent-checkpoint", type=str, default=None,
                        help="Path to an agent's checkpoint. Must match the master log.")
    return parser.parse_args()


def main(args: Namespace):
    target_date = datetime.fromisoformat(args.target_timestamp)
    load_checkpoint_and_eval(args.run_name, args.agent_name, target_date,
                             master_log_path=args.master_log,
                             agent_load_path=args.agent_checkpoint)


def load_checkpoint_and_eval(
    run_name: str, agent_name: str, target_date: datetime,
    master_log_path: Path | str = None, agent_load_path: Path = None,
) -> tuple[float, str]:

    # Load master log and fetch target user message
    if master_log_path is None:
        master_log_path = make_master_log_path(run_name, agent_name)
    master_log = MasterLog(master_log_path)
    master_log.load()
    llm_call_idx = 0
    target_message = None
    for evt in master_log.log:
        if evt.type in {EventType.RESPONSE_MESSAGE, EventType.RESPONSE_FILL}:
            llm_call_idx += 1
        if evt.timestamp < target_date:
            continue
        if evt.type not in {EventType.SEND_MESSAGE, EventType.SEND_FILL}:
            continue
        target_message = evt.data["message"]
        target_date = evt.timestamp
        break
    assert target_message is not None, f"Couldn't find message with timestamp {repr(target_date)}"
    colour_print("lightblue", "User message:")
    print(target_message)

    # `model` has "/" symbols removed, but that suffices for loading.
    model, prompt_size, message_size = agent_name.split(" - ")[-3:]
    prompt_size = int(prompt_size)
    message_size = int(message_size)
    agent = LTMAgentWrapper(run_name, model=model, max_prompt_size=prompt_size, max_message_size=message_size)
    agent.load(agent_load_path)
    agent.agent.llm_call_idx = llm_call_idx

    # Patch the retrieval, so that it cannot deliver future memories.
    def patched_retrieve(*args, **kwargs):
        memories = patched_retrieve.retrieve(*args, **kwargs)
        limit_timestamp = target_date.timestamp()
        return [m for m in memories if m.timestamp < limit_timestamp]

    patched_retrieve.retrieve = agent.agent.hybrid_memory.semantic_memory.retrieve
    agent.agent.hybrid_memory.semantic_memory.retrieve = patched_retrieve

    # Same for recent messages
    def patched_recent_messages(*args, **kwargs):
        messages = patched_recent_messages.get_recent_messages(*args, **kwargs)
        limit_timestamp = target_date.timestamp()
        return [m for m in messages if m.timestamp < limit_timestamp]

    patched_recent_messages.get_recent_messages = agent.agent.hybrid_memory.get_recent_messages
    agent.agent.hybrid_memory.get_recent_messages = patched_recent_messages

    # Travel to specific date and perform test
    with time_machine.travel(target_date.astimezone(timezone.utc)):
        colour_print("lightred", "Agent is running:")
        response, sent_dt, reply_dt = agent.message_to_agent(target_message)
        colour_print("lightred", "Agent response:")
        print(response)
        t_response = (reply_dt - sent_dt).total_seconds()
        colour_print("lightgreen", f"Response took {t_response:.2e} seconds and ${agent.costs_usd}")

    return t_response, response


def my_eval():
    run_name = "Dev Benchmark 2 32k - 2 Examples"
    agent_name = "LTMAgentWrapper - gpt-4o - 16384 - 1024"
    timestamps = [
        # Restaurant mix-up
        # Waiter: Here you are: Vegetarian Stir-Fry with Tofu, Mixed Vegetables, and Teriyaki Sauce over Steamed Rice. Enjoy the meal.
        "2024-09-03 23:02:43.262463",
        # SallyAnne 1
        # The TV program has ended for today. Where will Abigail look for the undershirt?
        "2024-09-03 23:16:14.043813",
        # Shopping 2
        # What is on my current shopping list?
        "2024-09-04 11:36:19.178763",
        # Trigger response 2 (control)
        # What is the current time?
        "2024-09-04 07:23:54.725328",
        # Name List 2 (control)
        # What have been all of the names that I have gone by? Express the answer as a JSON list.
        "2024-09-04 11:32:14.202532",
        # Jokes 2 (control)
        # Which joke did I tell you about 11 hours 58 minutes ago?
        "2024-09-04 11:25:47.179451",
    ]
    for ts in timestamps[:1]:
        target_date = datetime.fromisoformat(ts)
        t_response, response = load_checkpoint_and_eval(run_name, agent_name, target_date)
        p = DATA_DIR / "ckpt_eval" / f"{ts}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fd:
            json.dump(dict(response=response, time=t_response), fd, indent=2)
        with (DATA_DIR / "ckpt_eval" / f"{ts}.txt").open("w") as fd:
            fd.write(response)


if __name__ == "__main__":
    # main(get_args())
    my_eval()
