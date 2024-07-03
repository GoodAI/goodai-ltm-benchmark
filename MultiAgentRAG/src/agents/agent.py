import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import List, Tuple
from config import config
import json
from src.utils.api_utils import rate_limited, exponential_backoff, cached

logger = logging.getLogger('master')
chat_logger = logging.getLogger('chat')

class Agent:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.chat_model = ChatOpenAI(model_name=config.MODEL_NAME)

    # @cached
    # @rate_limited(max_calls=6, period=60)  # Adjust these values based on your API limits
    # @exponential_backoff(max_retries=3, base_delay=2)
    async def process_query(self, query: str) -> str:
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(query)
        
        messages = [
            HumanMessage(content=f"""REQUEST = "{query}". 

CONTEXT = "{relevant_memories}"

INSTRUCTIONS = The 'CONTEXT' provided contains relevant memories retrieved based on similarity to the current request. The format of the context is as follows:

Similar by [Similarity Metric] (ordered by timestamp - ascending):
<memory_id>, <previous_query>, <previous_response>, <timestamp>, <similarity_score>

For memories that have been mentioned before, the format is shortened to:
<memory_id>, <similarity_score>

Use this context to inform your response to the current request. Pay attention to the similarity metrics and scores to gauge the relevance of each memory. Memories with higher similarity scores and more recent timestamps may be more pertinent to the current query.

Please respond to the REQUEST based on this context and your general knowledge.""")
        ]
        
        # Log the full message sent to the API
        chat_logger.info(f"Full ChatGPT API request: {json.dumps([m.dict() for m in messages], indent=2)}")
        
        response = await self.chat_model.ainvoke(messages)
        
        # Log the full API response
        chat_logger.info(f"Full ChatGPT API response: {json.dumps(response.dict(), indent=2)}")
        
        logger.debug(f"Processed query: {query} with context: {relevant_memories}")
        
        return response.content