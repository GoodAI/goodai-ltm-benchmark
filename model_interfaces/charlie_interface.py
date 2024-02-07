import json
from dataclasses import dataclass, field
from typing import List

import browser_cookie3
from model_interfaces.interface import ChatSession
import requests


def try_extract_session_cookie(cj):
    user_name = ""
    session_token = ""
    for cookie in cj:
        if cookie.name == "session_token":
            session_token = cookie.value
        if cookie.name == "username":
            user_name = cookie.value.replace('"', "")
    return session_token, user_name


@dataclass
class CharlieMnemonic(ChatSession):
    context: List[str] = field(default_factory=list)
    max_prompt_size: int = 8192
    chat_id: str = "New chat"
    endpoint: str = "https://clang.goodai.com"
    token: str = ""
    user_name: str = ""
    initial_costs_usd: float = 0.0

    @property
    def name(self):
        return f"{super().name} - {self.max_prompt_size}"

    def __post_init__(self):
        browsers = [
            browser_cookie3.chrome,
            browser_cookie3.firefox,
            browser_cookie3.chromium,
            browser_cookie3.edge,
            browser_cookie3.safari,
        ]

        # Extract session token from cookie
        # We don't know which browser the user is using, so search for the most obvious ones:
        for b in browsers:
            try:
                cj = list(b(domain_name="clang.goodai.com"))
                self.token, self.user_name = try_extract_session_cookie(cj)
                if self.token != "" and self.user_name != "":
                    break
            except:
                continue

        if self.token == "":
            raise ValueError("No valid clang login found! Please login via browser.")

        body = {"username": self.user_name, "session_token": self.token}
        valid = requests.post(self.endpoint + "/check_token/", json=body)

        if valid.status_code != 200:
            raise ValueError(
                "Username/session token combination found by invalid! Please make sure your cookies are up to date."
            )

        # Get display name and current costs of user
        settings_dict = self.get_settings()
        self.display_name = settings_dict["display_name"][0]
        self.initial_costs_usd = settings_dict["usage"]["total_cost"]

        # Update max_tokens
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"session_token={self.token}",
        }
        body = {
            "username": self.user_name,
            "category": "memory",
            "setting": "max_tokens",
            "value": self.max_prompt_size,
        }

        update = requests.post(
            self.endpoint + "/update_settings/", headers=headers, json=body
        )

    def reply(self, user_message) -> str:
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"session_token={self.token}",
        }

        body = {
            "prompt": user_message,
            "username": self.user_name,
            "display_name": self.display_name,
            "chat_id": self.chat_id,
        }

        response_json = json.loads(
            requests.post(self.endpoint + "/message/", headers=headers, json=body).text
        )

        # Update costs
        settings = self.get_settings()
        self.costs_usd = settings["usage"]["total_cost"] - self.initial_costs_usd

        try:
            return response_json["content"]
        except KeyError as exc:
            exc.add_note(f"Received JSON:\n{response_json}")
            raise

    def get_settings(self):
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"session_token={self.token}",
        }
        body = {"username": self.user_name}

        settings = requests.post(
            self.endpoint + "/load_settings/", headers=headers, json=body
        )
        return json.loads(settings.text)

    def reset(self):
        # Delete the user data
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"session_token={self.token}",
        }
        body = {"username": self.user_name}
        delete_req = requests.post(
            self.endpoint + "/delete_data_keep_settings", headers=headers, json=body
        )

        self.context = []
