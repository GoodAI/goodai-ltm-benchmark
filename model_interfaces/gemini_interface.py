import os
import pickle
import google.generativeai as genai
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from model_interfaces.interface import ChatSession


@dataclass
class GeminiProInterface(ChatSession):
    is_local: bool = True  # It is free for now

    @property
    def history_path(self) -> Path:
        return self.save_path.joinpath("history.dat")

    def __post_init__(self):
        super().__post_init__()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        #self.model = genai.GenerativeModel('gemini-pro')
        self.reset()

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        return self.chat.send_message(user_message).text

    def reset(self, history=None):
        self.chat = self.model.start_chat(history=history or [])

    def save(self):
        with open(self.history_path, "bw") as fd:
            pickle.dump(self.chat.history, fd)

    def load(self):
        with open(self.history_path, "br") as fd:
            self.reset(pickle.load(fd))

    # def save(self):
    #     context = [dict(role=m.role, message=m.parts[0].text) for m in self.chat.history]
    #     with open(self.context_path, "w") as fd:
    #         json.dump(context, fd)
    #
    # def load(self):
    #     with open(self.context_path, "r") as fd:
    #         context = json.load(fd)
    #     history = [
    #         genai.generative_models.content_types.ContentDict(
    #             role=m["role"], parts=[m["message"]],
    #         ) for m in context
    #     ]
    #     self.reset(history=history)
