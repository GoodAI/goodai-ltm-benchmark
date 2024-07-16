import json
import re
from together import Together
from app.db.memory_manager import MemoryManager
from app.config import config
from app.utils.logging import get_logger

logger = get_logger(__name__)

class Agent:
    def __init__(self, api_key: str, memory_manager: MemoryManager):
        self.together_client = Together(api_key=api_key)
        self.memory_manager = memory_manager

    async def process_query(self, query: str) -> str:
        # Check if the query is a special trivia request
        if self._is_trivia_request(query):
            return await self._process_trivia_request(query)
        
        relevant_memories = await self._retrieve_relevant_memories(query)
        response = await self._generate_response(query, relevant_memories)
        await self._update_memory(query, response)
        return response

    def _is_trivia_request(self, query: str) -> bool:
        pattern = r'^\s*Here are some trivia questions and answers for you to process\.'
        return bool(re.match(pattern, query, re.IGNORECASE))

    async def _process_trivia_request(self, query: str) -> str:
        # Extract answers from the query
        answers = re.findall(r'\bAnswer:\s*(.+?)(?=\n|$)', query, re.IGNORECASE)
        return json.dumps(answers)

    async def _retrieve_relevant_memories(self, query: str) -> list:
        relevant_memories = await self.memory_manager.get_relevant_memories(query, top_k=5)
        return [memory[0] for memory in relevant_memories]  # Return only the content

    async def _generate_response(self, query: str, relevant_memories: list) -> str:
        prompt = self._construct_prompt(query, relevant_memories)
        response = self.together_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.MODEL_NAME,
            # max_tokens=1000  # Add this to limit the response length
        )
        return response.choices[0].message.content

    async def _update_memory(self, query: str, response: str):
        memory_id = await self.memory_manager.add_memory(f"Query: {query}\nResponse: {response}")
        logger.info(f"Memory updated with ID: {memory_id}")

    def _construct_prompt(self, query: str, relevant_memories: list) -> str:
        memory_context = "\n".join([f"- {memory}" for memory in relevant_memories])
        return f"""Given the following context and query, provide a relevant and informative response:

Context:
{memory_context}

Query: {query}

Response:"""