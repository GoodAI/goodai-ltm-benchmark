# multi_agent_interface.py
from typing import Optional
import requests
import os
from dotenv import load_dotenv
from model_interfaces.interface import ChatSession

load_dotenv()  # Load environment variables from .env file

class MultiAgentRAGInterface(ChatSession):
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.api_key = os.getenv('OPENAI_API_KEY')  # Fetch the API key from environment variables
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.costs_usd = 0  # Add an attribute to track costs

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"query": user_message}
        response = requests.post(f"{self.api_url}/query", json=payload, headers=headers)
        response.raise_for_status()
        # Example of incrementing costs (this should be replaced with actual cost calculation)
        self.costs_usd += 0.01
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

    def get_costs(self):
        # Method to return the current costs
        return self.costs_usd
