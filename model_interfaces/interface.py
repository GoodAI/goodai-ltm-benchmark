from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple
from dataclasses import dataclass

import tiktoken

from utils.constants import PERSISTENCE_DIR


@dataclass
class ChatSession(ABC):
    run_name: str = ""
    costs_usd: float = 0
    is_local: bool = False
    max_message_size: int = 1000

    def message_to_agent(self, user_message: str, agent_response: str = "") -> Tuple[str, datetime, datetime]:
        sent_ts = datetime.now()
        old_costs = self.costs_usd
        response = self.reply(user_message, agent_response)
        reply_ts = datetime.now()
        assert (
            self.is_local or old_costs < self.costs_usd
        ), "The agent implementation is not providing any cost information."
        return response, sent_ts, reply_ts

    def __post_init__(self):
        assert self.run_name != "", "Run name is not set!"

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def save_path(self):
        return PERSISTENCE_DIR.joinpath(self.save_name)

    @property
    def save_name(self):
        return f"{self.run_name} - {self.name}"

    @abstractmethod
    def reply(self, user_message: str, agent_response: str) -> str:
        """
        In this method, the agent is expected to:
        - Generate a response to "user_message" and return it as a plain string.
        - Update "costs_usd" with the costs incurred by the generation of the response.
          Not doing so will result in an error, unless the "is_local" flag is set.
        """
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    def token_len(self, text: str) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
