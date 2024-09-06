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

import time_machine
import argparse
from argparse import Namespace
from datetime import datetime, timezone
from model_interfaces.ltm_agent_wrapper import LTMAgentWrapper
from utils.files import make_master_log_path
from runner.master_log import MasterLog, EventType
from utils.ui import colour_print


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("agent_name", type=str)
    parser.add_argument("target_timestamp", type=str,
                        help="The timestamp of the user message that you want to test.")
    return parser.parse_args()


def main(args: Namespace):

    # Set the working time
    target_date = datetime.fromisoformat(args.target_timestamp)
    traveller = time_machine.travel(target_date.astimezone(timezone.utc))
    traveller.start()

    # Load master log and fetch target user message
    master_log_path = make_master_log_path(args.run_name, args.agent_name)
    master_log = MasterLog(master_log_path)
    master_log.load()
    target_message = None
    for evt in master_log.log:
        if evt.timestamp < target_date:
            continue
        if evt.type not in {EventType.SEND_MESSAGE, EventType.SEND_FILL}:
            continue
        target_message = evt.data["message"]
        break
    assert target_message is not None, f"Couldn't find message with timestamp {repr(target_date)}"

    # `model` has "/" symbols removed, but that suffices for loading.
    model, prompt_size, message_size = args.agent_name.split(" - ")[-3:]
    agent = LTMAgentWrapper(args.run_name, model=model, max_prompt_size=prompt_size, max_message_size=message_size)
    agent.load()

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

    # Send now the message and collect the agent's response
    colour_print("lightblue", "User message:")
    print(target_message)
    colour_print("lightred", "Agent is running:")
    response, sent_dt, reply_dt = agent.message_to_agent(target_message)
    colour_print("lightred", "Agent response:")
    print(response)
    t_response = (reply_dt - sent_dt).total_seconds()
    colour_print("lightgreen", f"Response took {t_response:.2e} seconds and ${agent.costs_usd}")


if __name__ == "__main__":
    main(get_args())
