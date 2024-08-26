# memorybank_interface.py
from typing import Optional
import requests
import os
from dotenv import load_dotenv
from model_interfaces.interface import ChatSession

load_dotenv()

class MemoryBankInterface(ChatSession):
    def __init__(self, api_url: str = "http://localhost:5000"):
        super().__init__()
        self.api_url = api_url
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.costs_usd = 0  # Initialize costs to 0 (in microdollars)
        self.username_set = False
        self.is_local = True

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"query": user_message}
        
        response = requests.post(f"{self.api_url}/query", json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        agent_response = response_data["response"]
        
        # Placeholder: Increment costs by a fixed amount for each query
        self.costs_usd += 1000  # Add 1000 microdollars ($0.001) per query
        
        return agent_response

    def reset(self):
        self.costs_usd = 0
        self.username_set = False

    def save(self):
        pass

    def load(self):
        pass

    def token_len(self, text: str) -> int:
        return len(text.split())

    def get_costs(self):
        return self.costs_usd
    
    def message_to_agent(self, user_message: str, agent_response: Optional[str] = None) -> tuple:
        old_costs = self.costs_usd
        response = self.reply(user_message, agent_response)
        new_costs = self.costs_usd
        return response, old_costs, new_costs