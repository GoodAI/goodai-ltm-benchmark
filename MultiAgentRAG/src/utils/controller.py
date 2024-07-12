import logging
from typing import List, Tuple
from src.agents.agent import Agent
from src.memory.memory_manager import MemoryManager
from config import config
from src.utils.logging_config import setup_logging
from src.utils.structured_logging import get_logger
import os

from src.utils.structured_logging import get_logger

class Controller:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.memory_manager = None
        self.agent = None

    async def initialize(self):
        container_id = os.environ.get('HOSTNAME', 'local')
        self.memory_manager = MemoryManager(config.OPENAI_API_KEY)  # Use OpenAI API key for embeddings
        await self.memory_manager.initialize()
        self.agent = Agent(self.memory_manager, config.GROQ_API_KEY)  # Pass Groq API key to Agent

    async def execute_query(self, query: str) -> str:
        try:
            response = await self.agent.process_query(query)
            self.logger.debug("Generated response", response=response)
            await self.memory_manager.save_memory(query, response)
            return response
        except Exception as e:
            self.logger.error(f"Error executing query", query=query, error=str(e), exc_info=True)
            return f"An error occurred while processing your query: {str(e)}"

    async def get_recent_memories(self, limit: int) -> List[Tuple[str, str]]:
        return await self.memory_manager.get_memories(limit)