import os
import json

from box.box import Box
from memgpt.agent import Agent

from memgpt.interface import CLIInterface as interface
import memgpt.system as system
import memgpt.constants as constants
from memgpt.constants import (
    LLM_MAX_TOKENS,
    CLI_WARNING_PREFIX,
    MESSAGE_SUMMARY_WARNING_FRAC,
)
from memgpt.config import AgentConfig, MemGPTConfig

import memgpt.utils as utils
from memgpt.openai_tools import is_context_overflow_error
from memgpt.utils import printd
from memgpt.presets import presets

from model_interfaces.interface import ChatSession
from memgpt.persistence_manager import LocalStateManager
from utils.openai import token_cost
import subprocess
from dataclasses import dataclass
from contextlib import contextmanager


MEMGPT_LOGS_FILE = "model_interfaces/memgpt-logs.jsonl"


def configure(context_length):
    model = "gpt-4"
    endpoint_type = "openai"
    proxy_endpoint = "http://localhost:5000/v1"
    embedding_dim = 1536
    context_length = min(context_length, LLM_MAX_TOKENS.get(model, context_length))

    if not MemGPTConfig.exists():
        MemGPTConfig.create_config_dir()
        config = MemGPTConfig(
            # model configs
            model=model,
            model_endpoint=proxy_endpoint,
            model_endpoint_type=endpoint_type,
            context_window=context_length,
            # embedding configs
            embedding_endpoint_type=endpoint_type,
            embedding_endpoint=proxy_endpoint,
            embedding_dim=embedding_dim,
            # credentials
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        config = MemGPTConfig.load()
        config.context_window = context_length
    config.save()

    agent_config = AgentConfig(
        persona=None,
        human=None,
        model=model,
        model_endpoint=proxy_endpoint,
        model_endpoint_type=endpoint_type,
        context_window=context_length,
        # embedding configs
        embedding_endpoint_type=endpoint_type,
        embedding_endpoint=proxy_endpoint,
        embedding_dim=embedding_dim,
        name="MemGPTAgent",  # It will use it for saving
    )
    agent_config.save()
    return agent_config


def create_memgpt_agent(context_length):
    agent_config = configure(context_length)
    # create agent
    memgpt_agent = presets.use_preset(
        agent_config.preset,
        agent_config,
        agent_config.model,
        utils.get_persona_text(agent_config.persona),
        utils.get_human_text(agent_config.human),
        interface,
        LocalStateManager(agent_config),
    )
    return memgpt_agent


def clear_cost_info():
    if os.path.exists(MEMGPT_LOGS_FILE):
        os.remove(MEMGPT_LOGS_FILE)


@contextmanager
def proxy(proxy_file_path: str):
    proc = None
    try:
        clear_cost_info()
        proc = subprocess.Popen(
            ["python", proxy_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Read a line to make sure it is active
        proc.stdout.readline()
        yield
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()


def read_cost_info() -> float:
    if not os.path.exists(MEMGPT_LOGS_FILE):
        return 0
    cost_usd = 0
    with open(MEMGPT_LOGS_FILE) as fd:
        for line in fd.readlines():
            d = json.loads(line)
            prompt_tokens = d["usage"]["prompt_tokens"]
            completion_tokens = d["usage"]["completion_tokens"]
            prompt_cost, completion_cost = token_cost(d["model"])
            cost_usd += prompt_tokens * prompt_cost
            cost_usd += completion_tokens * completion_cost
    os.remove(MEMGPT_LOGS_FILE)
    return cost_usd


@dataclass
class MemGPTChatSession(ChatSession):
    _max_prompt_size: int = 8192
    _proxy_path: str = "model_interfaces/memgpt_proxy.py"
    max_message_size: int = 300

    def __post_init__(self):
        self.reset()

    @property
    def max_prompt_size(self):
        return self._max_prompt_size

    @max_prompt_size.setter
    def max_prompt_size(self, value):
        self._max_prompt_size = value
        self.memgpt_agent.config.context_window = self._max_prompt_size

    def reset(self):
        # Create new memgpt agent
        self.memgpt_agent: Agent = create_memgpt_agent(self._max_prompt_size)
        self.memgpt_agent.__class__.step = step
        self.memgpt_agent._messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "I shall greet the user",
                    "function_call": {
                        "name": "send_message",
                        "arguments": '{\n "message": "Hello user how are you today?"\n}',
                    },
                },
                {
                    "role": "user",
                    "content": "I am well thank you! I am going to test you now and need to to always send a reply to me every time I speak to you.",
                },
                {
                    "role": "assistant",
                    "content": "How interesting! I will make sure to always use send_message when conversing with the the user.",
                    "function_call": {
                        "name": "send_message",
                        "arguments": '{\n "message": "That is very interesting! I will make sure to use send_message to reply to your tests. Lets begin!"\n}',
                    },
                },
            ]
        )

    @property
    def name(self):
        return f"{super().name} - {self.max_prompt_size}"

    # Stripped out version of MemGPT/memgpt/main.py:run_agent_loop()
    def reply(self, user_message: str) -> str:
        def process_agent_step(user_message):
            (
                new_messages,
                heartbeat_request,
                function_failed,
                token_warning,
            ) = self.memgpt_agent.step(user_message, first_message=False, skip_verify=True)

            skip_next_user_input = False
            if token_warning:
                user_message = system.get_token_limit_warning()
                skip_next_user_input = True
            elif function_failed:
                user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
                skip_next_user_input = True
            elif heartbeat_request:
                user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                skip_next_user_input = True

            return new_messages, user_message, skip_next_user_input

        with proxy(self._proxy_path):
            skip_next_user_input = True
            while skip_next_user_input:
                new_messages, user_message, skip_next_user_input = process_agent_step(user_message)
            messages = self.get_sent_messages(new_messages)

        self.costs_usd += read_cost_info()
        return "\n".join(messages)

    def get_sent_messages(self, message_list):
        message_strings = []

        for item in message_list:
            if type(item) is Box:
                if hasattr(item, "function_call"):
                    if item.function_call.name == "send_message":
                        arg_dict = json.loads(item.function_call.arguments)
                        message_strings.append(arg_dict["message"])

        return message_strings


# This is a copy of memgpt.step which deals with custom context sizes.
def step(
    self,
    user_message,
    first_message=False,
    first_message_retry_limit=constants.FIRST_MESSAGE_ATTEMPTS,
    skip_verify=False,
):
    """Top-level event message handler for the MemGPT agent"""

    try:
        # Step 0: add user message
        if user_message is not None:
            self.interface.user_message(user_message)
            packed_user_message = {"role": "user", "content": user_message}
            input_message_sequence = self.messages + [packed_user_message]
        else:
            input_message_sequence = self.messages

        if len(input_message_sequence) > 1 and input_message_sequence[-1]["role"] != "user":
            printd(
                f"{constants.CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue"
            )

        # Step 1: send the conversation and available functions to GPT
        if not skip_verify and (first_message or self.messages_total == self.messages_total_init):
            printd(f"This is the first message. Running extra verifier on AI response.")
            counter = 0
            while True:
                response = self.get_ai_reply(
                    message_sequence=input_message_sequence,
                )
                if self.verify_first_message_correctness(response, require_monologue=self.first_message_verify_mono):
                    break

                counter += 1
                if counter > first_message_retry_limit:
                    raise Exception(f"Hit first message retry limit ({first_message_retry_limit})")

        else:
            response = self.get_ai_reply(
                message_sequence=input_message_sequence,
            )

        # Step 2: check if LLM wanted to call a function
        # (if yes) Step 3: call the function
        # (if yes) Step 4: send the info on the function call and function response to LLM
        response_message = response.choices[0].message
        response_message_copy = response_message.copy()
        all_response_messages, heartbeat_request, function_failed = self.handle_ai_response(response_message)

        # Add the extra metadata to the assistant response
        # (e.g. enough metadata to enable recreating the API call)
        assert "api_response" not in all_response_messages[0]
        all_response_messages[0]["api_response"] = response_message_copy
        assert "api_args" not in all_response_messages[0]
        all_response_messages[0]["api_args"] = {
            "model": self.model,
            "messages": input_message_sequence,
            "functions": self.functions,
        }

        # Step 4: extend the message history
        if user_message is not None:
            all_new_messages = [packed_user_message] + all_response_messages
        else:
            all_new_messages = all_response_messages

        # Check the memory pressure and potentially issue a memory pressure warning
        current_total_tokens = response["usage"]["total_tokens"]
        active_memory_warning = False

        if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * int(self.config.context_window):
            printd(
                f"{CLI_WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * int(self.config.context_window)}"
            )
            # Only deliver the alert if we haven't already (this period)
            if not self.agent_alerted_about_memory_pressure:
                active_memory_warning = True
                self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this

        # This is what is added
        if current_total_tokens > int(self.config.context_window):
            printd("Summarising messages")
            self.summarize_messages_inplace()

        else:
            printd(
                f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * int(self.config.context_window)}"
            )

        self.append_to_messages(all_new_messages)
        return (
            all_new_messages,
            heartbeat_request,
            function_failed,
            active_memory_warning,
        )

    except Exception as e:
        printd(f"step() failed\nuser_message = {user_message}\nerror = {e}")

        # If we got a context alert, try trimming the messages length, then try again
        if is_context_overflow_error(e):
            # A separate API call to run a summarizer

            self.summarize_messages_inplace()
            # Try step again
            return self.step(user_message, first_message=first_message)
        else:
            printd(f"step() failed with an unrecognized exception: '{str(e)}'")
            raise e


if __name__ == "__main__":
    dd = MemGPTChatSession()
    a = dd.message_to_agent("Hello")
