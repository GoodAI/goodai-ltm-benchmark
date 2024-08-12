import time

import time_machine
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional
from dataclasses import dataclass
from utils.constants import PERSISTENCE_DIR
from utils.llm import count_tokens_for_model


@dataclass
class ChatSession(ABC):
    run_name: str = ""
    costs_usd: float = 0
    is_local: bool = False
    max_message_size: int = 4096
    time_travel: bool = True
    traveller: Optional[time_machine.travel] = None

    def message_to_agent(self, user_message: str, agent_response: Optional[str] = None) -> Tuple[str, datetime, datetime]:
        sent_ts = datetime.now()
        old_costs = self.costs_usd
        response = self.reply(user_message, agent_response=agent_response)
        reply_ts = datetime.now()
        # If we are supplying a response from the agent, then don't count costs.
        if agent_response is None:
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
    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
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
        return count_tokens_for_model(text=text)

    def forward_time(self, seconds: float):

        if not self.time_travel:
            print(f"Time travels are deactivated. Waiting for {seconds} seconds.")
            while seconds > 0:
                print(f"\r{seconds} seconds left.     ", end="")
                time.sleep(min(seconds, 1))
                seconds -= 1
            print("\rWait ended.")
            return

        t_jump = timedelta(seconds=seconds)
        target_date = datetime.now() + t_jump
        assert target_date > datetime.now(), "Can only move forward in time. Going back is problematic."
        self.reset_time()
        self.traveller = time_machine.travel(target_date.astimezone(timezone.utc))
        self.traveller.start()

    def reset_time(self):
        if self.traveller is not None:
            self.traveller.stop()
            self.traveller = None
