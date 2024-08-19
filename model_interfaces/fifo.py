import os
import json
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from model_interfaces.interface import ChatSession


@dataclass
class FifoAgentInterface(ChatSession):
    fifo_file: Path = None

    def __post_init__(self):
        # Wait for the agent to be ready
        assert self.fifo_file is not None
        if not self.fifo_file.exists():
            os.mkfifo(self.fifo_file)
        assert self.fifo_file.is_fifo()
        self.is_local = self.receive()["is_local"]

    def __del__(self):
        self.send(method="end")

    def receive(self) -> dict:
        with open(self.fifo_file) as fd:
            return json.load(fd)

    def send(self, **kwargs):
        with open(self.fifo_file, "w") as fd:
            json.dump(kwargs, fd)

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        self.send(
            method="reply", user_message=user_message, agent_response=agent_response,
        )
        response = self.receive()
        self.costs_usd += response.get("cost_usd", 0)
        return response["return_value"]

    def token_len(self, text: str) -> int:
        self.send(method="token_len", text=text)
        return self.receive()["return_value"]

    def reset(self):
        self.send(method="reset")

    def save(self):
        self.send(method="save")

    def load(self):
        self.send(method="load")
