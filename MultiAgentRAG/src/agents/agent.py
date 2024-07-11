import logging
from groq import Groq # type: ignore
from config import config
import json
import asyncio

logger = logging.getLogger("master")
chat_logger = logging.getLogger("chat")

class Agent:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)

    async def process_query(self, query: str) -> str:
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(query)
        
        messages = [
            {
                "role": "system",
                "content": """You are a highly efficient, no-nonsense administrator at the core of a sophisticated memory system. Your primary role is to analyze retrieved memories and provide concise, relevant summaries or indicate their lack of relevance. You value brevity and directness above all else.

    The retrieved memories you will be processing are structured as follows:

    1. Each memory entry contains: <memory_id>, <previous_query>, <previous_response>, <similarity_score>
    2. Some entries may have an additional field indicating if they are linked to another memory.

    Analyze these memories in relation to the current query efficiently, paying special attention to linked memories."""
            },
            {
                "role": "user",
                "content": f"""QUERY: {query}

    RETRIEVED MEMORIES:
    {relevant_memories}

    Your task:

    1. Assess if the retrieved memories contain ANY relevant information to the query.
    2. If NO relevant information exists, respond ONLY with: "NO RELEVANT INFORMATION"
    3. If relevant information exists, provide a brief, bullet-point summary of ONLY the most pertinent points. Be extremely concise.
    4. Pay special attention to linked memories and their relevance to the current query.

    Your response should either be "NO RELEVANT INFORMATION" or a brief, focused summary. Exclude any explanation or elaboration."""
            }
        ]

        # Use asyncio.to_thread to run the synchronous API call in a separate thread
        response = await asyncio.to_thread(
            self.groq_client.chat.completions.create,
            messages=messages,
            model=config.MODEL_NAME,
        )

        return response.choices[0].message.content