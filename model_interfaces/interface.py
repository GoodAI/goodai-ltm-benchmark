from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple, Callable
from dataclasses import dataclass, field

from utils.constants import PERSISTENCE_DIR, ResetPolicy


@dataclass
class ChatSession(ABC):
    run_name: str = ""
    history: list[dict[str, str | datetime]] = field(default_factory=list)
    costs_usd: float = 0
    is_local: bool = False
    max_message_size: int = 1000
    reset_policy: ResetPolicy = ResetPolicy.SOFT

    def message_to_agent(self, user_message: str) -> Tuple[str, datetime, datetime]:
        sent_ts = datetime.now()
        self.history.append({"role": "user", "content": user_message, "timestamp": sent_ts})
        old_costs = self.costs_usd
        response = self.reply(user_message)
        reply_ts = datetime.now()
        assert (
            self.is_local or old_costs < self.costs_usd
        ), "The agent implementation is not providing any cost information."
        self.history.append({"role": "assistant", "content": response, "timestamp": reply_ts})
        return response, sent_ts, reply_ts

    def __post_init__(self):
        assert self.run_name != "", "Run name is not set!"
        assert isinstance(self.history, list), "History attribute not set!"

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
    def reply(self, user_message: str) -> str:
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

    def reset_history(self):
        # TODO This is meant for clearing the current conversation's history.
        # If 'history' needs to be consistent with cost, reset_history() should be abstract.
        self.history = []

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class ChatSessionFactory(ABC):
    def create_session(self) -> ChatSession:
        pass
