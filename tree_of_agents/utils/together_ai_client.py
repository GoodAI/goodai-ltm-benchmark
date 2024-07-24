import os
from together import Together

class TogetherAIClient:
    def __init__(self, model):
        self.client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
        self.model = model

    def generate_response(self, messages, max_tokens=32768, temperature=0):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content