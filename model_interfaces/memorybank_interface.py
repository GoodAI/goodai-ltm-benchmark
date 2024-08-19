# memorybank_interface.py
from typing import Optional
import requests
import os
from dotenv import load_dotenv
from model_interfaces.interface import ChatSession
import time
import uuid

load_dotenv()

class MemoryBankInterface(ChatSession):
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.costs_usd = 0
        self.username_set = False
        self.is_local = True
        self.run_name: str = "MemoryBankAPI"
        self.user_id = f"BenchUser_{uuid.uuid4().hex[:8]}"  # Generate a unique user ID

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"query": user_message, "user_id": self.user_id}

        if not self.username_set:
            self.username_set = True
            # Send initialization request
            init_response = requests.post(f"{self.api_url}/initialize", json=payload, headers=headers)
            init_response.raise_for_status()
            time.sleep(30)  # Add a 30-second delay after initialization

        # Send the actual query
        response = requests.post(f"{self.api_url}/query", json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        agent_response = response_data.get("response", "No response")
        
        self.costs_usd += 1000  # Add 1000 microdollars ($0.001) per query
        
        return agent_response

    def reset(self):
        requests.post(f"{self.api_url}/reset")
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
