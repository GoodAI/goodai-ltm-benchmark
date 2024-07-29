from openai import OpenAI
from config import OPENAI_API_KEY

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_response(self, messages, model="gpt-4-turbo-preview", max_tokens=150):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")