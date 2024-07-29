from typing import Dict, Any
from openai import OpenAI
from groq import Groq
from together import Together
from app.config import config
from app.utils.logging import get_logger

logger = get_logger('custom')

class ModelClient:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == 'openai':
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        elif provider == 'groq':
            self.client = Groq(api_key=config.GROQ_API_KEY)
        elif provider == 'together':
            self.client = Together(api_key=config.TOGETHER_API_KEY)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat_completion(self, model: str, messages: list, **kwargs) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
            raise

    def get_completion_content(self, completion: Dict[str, Any]) -> str:
        return completion.choices[0].message.content