import logging
from typing import List, Tuple
from src.agents.agent import Agent
from src.memory.memory_manager import MemoryManager
from config import config

class Controller:
    def __init__(self):
        self.logger = logging.getLogger('master')
        self.db_logger = logging.getLogger('database')
        self.memory_manager = None
        self.agent = None

    async def initialize(self):
        self.memory_manager = MemoryManager(config.OPENAI_API_KEY)
        await self.memory_manager.initialize()
        self.agent = Agent(self.memory_manager)

    async def execute_query(self, query: str) -> str:
        try:
            response = await self.agent.process_query(query)
            self.logger.debug(f"Generated response: {response}")
            await self.memory_manager.save_memory(query, response)
            return response
        except Exception as e:
            self.logger.error(f"Error executing query '{query}': {str(e)}", exc_info=True)
            raise

    async def get_recent_memories(self, limit: int) -> List[Tuple[str, str]]:
        return await self.memory_manager.get_memories(limit)