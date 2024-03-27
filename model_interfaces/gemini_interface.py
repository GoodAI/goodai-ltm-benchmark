import os
import google.generativeai as genai
import json
from model_interfaces.interface import ChatSession
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeminiChatSession(ChatSession):
    model_name: str = "gemini-pro"

    def __post_init__(self):
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.reset()

    @property
    def name(self) -> str:
        return f"{super().name} - {self.model_name}"

    @property
    def context_path(self) -> str:
        return self.save_path.joinpath("context.json")

    def reply(self, user_message: str) -> str:
        response = self.chat.send_message(user_message)
        return response.text

    def reset(self, history: Optional[list] = None):
        self.model = genai.GenerativeModel('gemini-pro')
        self.chat = self.model.start_chat(history=history)

    def save(self):
        context = [dict(role=m.role, message=m.parts[0].text) for m in self.chat.history]
        with open(self.context_path, "w") as fd:
            json.dump(context, fd)

    def load(self):
        with open(self.context_path, "r") as fd:
            context = json.load(fd)
        history = [
            genai.generative_models.content_types.ContentDict(
                role=m["role"], parts=[m["message"]],
            ) for m in context
        ]
        self.reset(history=history)
