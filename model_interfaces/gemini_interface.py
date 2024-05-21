import os
import time
import pickle
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.ai import generativelanguage as glm
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
        self.reset()

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        if agent_response is not None:
            self.chat.history.append(glm.Content({'role': 'user', 'parts': [glm.Part({"text": user_message})]}))
            self.chat.history.append(glm.Content({'role': 'model', 'parts': [glm.Part({"text": agent_response})]}))
            return agent_response
        response = self.chat.send_message(
            user_message,
            safety_settings={
                cat: HarmBlockThreshold.BLOCK_NONE
                for cat in [
                    HarmCategory.HARM_CATEGORY_HARASSMENT,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                ]
            }
        ).text
        time.sleep(30)  # 2 requests per minute in the free tier
        return response

    def reset(self, history=None):
        self.chat = self.model.start_chat(history=history or [])

    def save(self):
        with open(self.history_path, "bw") as fd:
            pickle.dump(self.chat.history, fd)

    def load(self):
        with open(self.history_path, "br") as fd:
            self.reset(pickle.load(fd))

    def token_len(self, text: str) -> int:
        for i in range(3):
            try:
                return self.model.count_tokens(text, request_options={"timeout": 10000}).total_tokens
            except:  # google.api_core.exceptions.DeadlineExceeded:
                if i == 2:
                    raise
                time.sleep(5)
