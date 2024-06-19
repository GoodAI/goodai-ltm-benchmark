# multi_agent_interface.py
from typing import Optional

import requests
from model_interfaces.interface import ChatSession

class MultiAgentRAGInterface(ChatSession):
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        payload = {"query": user_message}
        response = requests.post(f"{self.api_url}/query", json=payload)
        response.raise_for_status()
        return response.json()["response"]

    def reset(self):
        # Implement reset functionality if needed
        pass

    def save(self):
        # Implement save functionality if needed
        pass

    def load(self):
        # Implement load functionality if needed
        pass

    def token_len(self, text: str) -> int:
        # Implement token length calculation if needed
        return len(text.split())
