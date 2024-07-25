import requests
from config import TOGETHER_API_KEY

class TogetherAIClient:
    def __init__(self, model):
        self.model = model
        self.api_url = "https://api.together.xyz/inference"
        self.headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

    def generate_response(self, messages):
        prompt = self.format_prompt(messages)
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 4096,
        }
        response = requests.post(self.api_url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()['output']['choices'][0]['text'].strip()

    def format_prompt(self, messages):
        formatted_messages = []
        for message in messages:
            if message["role"] == "system":
                formatted_messages.append(f"System: {message['content']}")
            elif message["role"] == "user":
                formatted_messages.append(f"Human: {message['content']}")
            elif message["role"] == "assistant":
                formatted_messages.append(f"Assistant: {message['content']}")
        return "\n".join(formatted_messages) + "\nAssistant:"