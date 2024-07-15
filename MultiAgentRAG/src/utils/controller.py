from typing import List, Tuple
from src.agents.agent import Agent
from src.memory.memory_manager import MemoryManager
from config import config
from src.utils.structured_logging import get_logger
from src.utils.error_handling import log_error_with_traceback
import os

class Controller:
    def __init__(self):
        self.logger = get_logger('master')
        self.memory_manager = None
        self.agent = None

    async def initialize(self):
        container_id = os.environ.get('HOSTNAME', 'local')
        self.memory_manager = MemoryManager(config.OPENAI_API_KEY)  # Use OpenAI API key for embeddings
        await self.memory_manager.initialize()
        # self.agent = Agent(self.memory_manager, config.GROQ_API_KEY)  # Pass Groq API key to Agent
        self.agent = Agent(self.memory_manager, config.TOGETHER_API_KEY)  #? Together version

    async def execute_query(self, query: str) -> str:
        try:
            self.logger.info("Executing query", extra={"query": query})
            response = await self.agent.process_query(query)
            self.logger.info("Query executed successfully", extra={"query": query})
            await self.memory_manager.save_memory(query, response)
            return response
        except Exception as e:
            log_error_with_traceback(self.logger, "Error executing query", e)
            return f"An error occurred while processing your query: {str(e)}"

    async def get_recent_memories(self, limit: int) -> List[Tuple[str, str]]:
        return await self.memory_manager.get_memories(limit)