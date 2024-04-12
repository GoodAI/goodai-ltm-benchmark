import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field

from memgpt.client.client import LocalClient
from memgpt.data_types import AgentState

from model_interfaces.interface import ChatSession


from memgpt import create_client

from utils.llm import token_cost

MEMGPT_LOGS_FILE = "model_interfaces/memgpt-logs.jsonl"


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


def clear_cost_info():
    if os.path.exists(MEMGPT_LOGS_FILE):
        os.remove(MEMGPT_LOGS_FILE)


def read_cost_info() -> float:
    if not os.path.exists(MEMGPT_LOGS_FILE):
        return 0
    cost_usd = 0
    with open(MEMGPT_LOGS_FILE) as fd:
        for line in fd.readlines():
            d = json.loads(line)
            prompt_tokens = d["usage"]["prompt_tokens"]
            completion_tokens = d["usage"].get("completion_tokens", 0)
            prompt_cost, completion_cost = token_cost(d["model"])
            cost_usd += prompt_tokens * prompt_cost
            cost_usd += completion_tokens * completion_cost
    os.remove(MEMGPT_LOGS_FILE)
    return cost_usd


@dataclass
class MemGPTChatSession(ChatSession):
    _proxy_path: str = "model_interfaces/memgpt_proxy.py"
    client: LocalClient = field(default_factory=create_client)
    agent_info: AgentState = None
    agent_name: str = "LTMBenchmarkAgent"
    agent_initialised: bool = False
    max_prompt_size: int = None

    def __post_init__(self):
        self.client = create_client()
        self.client.server.server_llm_config.model_endpoint = "http://localhost:5000/v1"
        self.client.server.server_embedding_config.embedding_endpoint = "http://localhost:5000/v1"
        if self.max_prompt_size is None:
            self.max_prompt_size = self.client.server.server_llm_config.context_window

        self.client.server.server_llm_config.context_window = self.max_prompt_size

    def reply(self, user_message: str, agent_response: str) -> str:
        if not self.agent_initialised:
            self.reset()

        with proxy(self._proxy_path):
            response = self.client.user_message(agent_id=self.agent_info.id, message=user_message)

        messages = []
        for res_dict in response:
            if "assistant_message" in res_dict:
                messages.append(res_dict["assistant_message"])

        self.costs_usd += read_cost_info()
        return "\n".join(messages)

    def agent_id_from_name(self, name: str):
        if not self.client.agent_exists(agent_name=name):
            return None

        agent_dict = self.client.server.list_agents(self.client.user_id)
        for a in agent_dict["agents"]:
            if a["name"] == name:
                return a["id"]

    def init_agent(self):
        self.agent_initialised = True
        if not self.client.agent_exists(agent_name=self.agent_name):
            self.agent_info = self.client.create_agent(name=self.agent_name)
        else:
            self.agent_info = self.client.get_agent_config(self.agent_id_from_name(self.agent_name))

    def reset(self):
        self.agent_initialised = True
        self.client.server.delete_agent(self.client.user_id, self.agent_id_from_name(self.agent_name))
        self.agent_info = self.client.create_agent(name=self.agent_name)
        self.save()

    def save(self):
        self.client.save()

    def load(self):
        self.init_agent()


if __name__ == "__main__":
    dd = MemGPTChatSession()
    a = dd.message_to_agent("Hello")
