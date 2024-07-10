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
        # Check for trivia indicator
        trivia_indicator = "Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message:"
        
        if trivia_indicator in query:
            logger.info("Trivia question detected. Skipping processing and API call.")
            return "Deemed trivia - not added to DB"

        return await self._process_non_trivia_query(query)

    async def _process_non_trivia_query(self, query: str) -> str:
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(query)
        
        messages = [
            {
                "role": "system",
                "content": """You are a highly efficient, no-nonsense administrator at the core of a sophisticated memory system. Your primary role is to analyze retrieved memories and provide concise, relevant summaries or indicate their lack of relevance. You value brevity and directness above all else.

The retrieved memories you will be processing are structured as follows:

1. Memories are grouped by similarity metric (e.g., "Similar by L2 norm", "Similar by Cosine Similarity", etc.).
2. Each group is presented in ascending order of timestamp.
3. Each memory entry contains: <memory_id>, <previous_query>, <previous_response>, <timestamp>, <similarity_score>
4. For previously mentioned memories in other groups: <memory_id>, <similarity_score>

Analyze these memories in relation to the current query efficiently."""
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

Your response should either be "NO RELEVANT INFORMATION" or a brief, focused summary. Exclude any explanation or elaboration."""
            }
        ]

        # Log the full message sent to the API
        chat_logger.info(f"Full Groq API request: {json.dumps(messages, indent=2)}")

        # Use asyncio.to_thread to run the synchronous API call in a separate thread
        response = await asyncio.to_thread(
            self.groq_client.chat.completions.create,
            messages=messages,
            model=config.MODEL_NAME,
        )

        # Log the full API response
        chat_logger.info(f"Full Groq API response: {json.dumps(response.model_dump(), indent=2)}")

        logger.debug(f"Processed query: {query} with context: {relevant_memories}")

        return response.choices[0].message.content