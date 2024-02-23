import sys
from datetime import datetime
from dataclasses import dataclass

from utils.constants import ResetPolicy
from utils.ui import colour_print, multiline_input
from model_interfaces.interface import ChatSession


@dataclass
class HumanChatSession(ChatSession):
    is_local: bool = True
    max_message_size: int = sys.maxsize


    def __post_init__(self):
        print(
            "This is a human interface for the GoodAI LTM Benchmark. You are expected to provide responses to the "
            "different tests' messages, as if you were an AI agent. Multi-line responses are allowed, and an empty line "
            "signals the end of the response."
        )

    def reset(self):
        print(
            "The tester now expects the agent to remember nothing from past interactions. We recommend you to switch "
            "places with another human, but if that is not possible, we ask you to not consider past information whilst "
            "answering the next questions."
        )

    def reply(self, user_message: str) -> str:
        colour_print("red", datetime.now().isoformat())
        colour_print("cyan", f"Test: {user_message}")
        return multiline_input("Human: ")

    def save(self):
        pass

    def load(self):
        pass