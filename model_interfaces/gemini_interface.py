import os
import time
import pickle
import requests
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.ai import generativelanguage as glm
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from model_interfaces.interface import ChatSession


def count_tokens_by_curl(text: str) -> int:
    api_key = os.getenv('GOOGLE_API_KEY')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:countTokens?key={api_key}'
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps({"contents": [{"parts": [{"text": text}]}]}))
    return r.json()["totalTokens"]


def reply_by_curl(history: list[glm.Content]) -> str:
    api_key = os.getenv('GOOGLE_API_KEY')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}'
    headers = {'Content-Type': 'application/json'}
    contents = [{"role": msg.role, "parts": [{"text": p.text} for p in msg.parts]} for msg in history]
    safety = [
        {"category": f"HARM_CATEGORY_{cat}", "threshold": "BLOCK_NONE"}
        for cat in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
    ]
    gen_config = {"temperature": 0.0}
    data = {"contents": contents, "safetySettings": safety, "generationConfig": gen_config}
    r = requests.post(url, headers=headers, data=json.dumps(data)).json()
    if "error" in r and r["error"]["code"] == 503:  # The model is overloaded. Please try again later.
        print(r)
        time.sleep(10)
        return reply_by_curl(history)
    try:
        return r["candidates"][0]["content"]["parts"][0]["text"]
    except:
        print(r)
        raise


@dataclass
class GeminiProInterface(ChatSession):
    is_local: bool = True  # TODO: Capture cost information

    @property
    def history_path(self) -> Path:
        return self.save_path.joinpath("history.dat")

    def __post_init__(self):
        super().__post_init__()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.reset()

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        self.chat.history.append(glm.Content({'role': 'user', 'parts': [glm.Part({"text": user_message})]}))
        if agent_response is None:
            agent_response = reply_by_curl(self.chat.history)
        self.chat.history.append(glm.Content({'role': 'model', 'parts': [glm.Part({"text": agent_response})]}))
        time.sleep(0.1)
        return agent_response

    def reset(self, history=None):
        self.chat = self.model.start_chat(history=history or [])

    def save(self):
        with open(self.history_path, "bw") as fd:
            pickle.dump(self.chat.history, fd)

    def load(self):
        with open(self.history_path, "br") as fd:
            self.reset(pickle.load(fd))

    def token_len(self, text: str) -> int:
        tokens = count_tokens_by_curl(text)
        time.sleep(0.1)
        return tokens
