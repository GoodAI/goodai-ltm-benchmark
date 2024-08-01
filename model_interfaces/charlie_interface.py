import json
from dataclasses import dataclass
from typing import Optional

import requests
from model_interfaces.interface import ChatSession


@dataclass
class CharlieMnemonic(ChatSession):
    max_prompt_size: int = 4160  # match with input_tokens in update_memory_settings
    chat_id: str = "Benchmark Chat"
    endpoint: str = "http://localhost:8002"  # default local endpoint
    username: str = "admin"  # default username
    password: str = "admin"  # default password
    initial_costs_usd: float = 0.0
    session: requests.Session = None
    # Adding this to the system prompt, because charlie tends to ask the user if it should save information instead of just saving it
    system_prompt: str = (
        "You are answering questions for an automated benchmark, don't ask the user if you should save information, just save it."
    )

    @property
    def name(self):
        return f"{super().name} - {self.max_prompt_size}"

    def __post_init__(self):
        super().__post_init__()
        self.session = requests.Session()
        self.login()
        self.create_chat_tab()
        self.update_memory_settings()
        self.update_max_tokens()
        self.load_initial_settings()
        self.update_system_prompt(self.system_prompt)

    def login(self):
        login_url = f"{self.endpoint}/login/"
        login_data = {"username": self.username, "password": self.password}
        response = self.session.post(login_url, json=login_data)
        if response.status_code != 200:
            raise ValueError("Login failed. Please check your credentials.")

    def update_system_prompt(self, new_system_prompt: str):
        update_settings_url = f"{self.endpoint}/update_settings/"
        update_data = {
            "username": self.username,
            "category": "system_prompt",
            "setting": {"system_prompt": new_system_prompt},
        }
        response = self.session.post(update_settings_url, json=update_data)
        if response.status_code != 200:
            raise ValueError("Failed to update system prompt setting.")
        self.system_prompt = new_system_prompt

    def create_chat_tab(self):
        create_chat_tab_url = f"{self.endpoint}/create_chat_tab/"
        chat_tab_data = {
            "username": self.username,
            "chat_id": self.chat_id,
            "chat_name": f"{self.chat_id}",
        }
        response = self.session.post(create_chat_tab_url, json=chat_tab_data)
        if response.status_code != 200:
            raise ValueError(f"Failed to create chat tab '{self.chat_id}'.")

    def update_max_tokens(self):
        update_settings_url = f"{self.endpoint}/update_settings/"
        update_data = {
            "username": self.username,
            "category": "memory",
            "setting": {"max_tokens": self.max_prompt_size},
        }
        response = self.session.post(update_settings_url, json=update_data)
        if response.status_code != 200:
            raise ValueError("Failed to update max_tokens setting.")

    # Charlie mnemonic has a lot of memory settings that need to be updated, performance varies greatly based on these settings
    def update_memory_settings(self):
        update_settings_url = f"{self.endpoint}/update_settings/"
        memory_settings = {
            "functions": 500,
            "ltm1": 960,  # active memory
            "ltm2": 960,  # category memory
            "episodic": 960,  # episodic memory
            "recent": 2380,  # recent messages
            "notes": 2080,  # notes/scratchpad
            "input": 4160,  # input tokens
            "output": 4000,  # output tokens
            "max_tokens": 16000,  # max tokens (all of the above combined)
            "min_tokens": 500,
        }
        update_data = {
            "username": self.username,
            "category": "memory",
            "setting": memory_settings,
        }
        response = self.session.post(update_settings_url, json=update_data)
        if response.status_code != 200:
            raise ValueError("Failed to update memory settings.")

    def load_initial_settings(self):
        load_settings_url = f"{self.endpoint}/load_settings/"
        response = self.session.post(load_settings_url)
        if response.status_code == 200:
            settings = response.json()
            self.display_name = settings.get("display_name", [self.username])[0]
            self.initial_costs_usd = settings.get("usage", {}).get("total_cost", 0.0)
        else:
            raise ValueError("Failed to load initial settings.")

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        message_url = f"{self.endpoint}/message/"
        message_data = {
            "prompt": user_message,
            "display_name": self.display_name,
            "chat_id": self.chat_id,
            "username": self.username,
        }
        response = self.session.post(message_url, json=message_data)
        if response.status_code == 200:
            response_json = response.json()
            self.update_costs()
            return response_json["content"]
        else:
            raise ValueError("Failed to send message.")

    def update_costs(self):
        load_settings_url = f"{self.endpoint}/load_settings/"
        response = self.session.post(load_settings_url)
        if response.status_code == 200:
            settings = response.json()
            current_cost = settings.get("usage", {}).get("total_cost", 0.0)
            self.costs_usd = current_cost - self.initial_costs_usd
        else:
            raise ValueError("Failed to update costs.")

    def reset(self):
        delete_data_url = f"{self.endpoint}/delete_data_keep_settings/"
        response = self.session.post(delete_data_url)
        if response.status_code != 200:
            raise ValueError("Failed to reset user data.")

    def load(self):
        # Charlie mnemonic is web based and so doesn't need to be manually told to resume a conversation
        # Use the chat_id to continue a conversation
        pass

    def save(self):
        # Charlie mnemonic is web based and so doesn't need to be manually told to persist
        pass
