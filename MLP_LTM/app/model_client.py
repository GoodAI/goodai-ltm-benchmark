import aiohttp
import asyncio
from openai import AsyncOpenAI
from groq import AsyncGroq
from together import Together
from app.config import config
from app.utils.logging import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = get_logger('custom')

class ModelClient:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == 'openai':
            self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        elif provider == 'groq':
            self.client = AsyncGroq(api_key=config.GROQ_API_KEY)
        elif provider == 'together':
            self.client = Together(api_key=config.TOGETHER_API_KEY)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def chat_completion(self, model: str, messages: list, **kwargs):
        try:
            if self.provider in ['openai', 'groq']:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
            elif self.provider == 'together':
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=messages,
                    **kwargs
                )
            logger.debug(f"API Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
            raise

    def get_completion_content(self, completion):
        logger.debug(f"Completion object: {completion}")
        
        if isinstance(completion, dict):
            if 'choices' in completion and len(completion['choices']) > 0:
                if 'message' in completion['choices'][0]:
                    return completion['choices'][0]['message'].get('content', '')
                elif 'text' in completion['choices'][0]:
                    return completion['choices'][0]['text']
        elif hasattr(completion, 'choices') and len(completion.choices) > 0:
            if hasattr(completion.choices[0], 'message'):
                return completion.choices[0].message.content
            elif hasattr(completion.choices[0], 'text'):
                return completion.choices[0].text
        
        logger.error(f"Unexpected completion format: {completion}")
        raise ValueError(f"Unexpected completion format: {completion}")

    def get_tokens_used(self, completion):
        if isinstance(completion, dict) and 'usage' in completion:
            return completion['usage'].get('total_tokens', 0)
        return 0

    async def aclose(self):
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()
