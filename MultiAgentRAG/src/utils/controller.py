import logging
from typing import List, Tuple
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.processing_agent import ProcessingAgent
from src.memory.memory_manager import MemoryManager
from config import config

class Controller:
    def __init__(self):
        self.memory_manager = MemoryManager(config.MEMORY_DB_PATH, config.OPENAI_API_KEY)
        self.retrieval_agent = RetrievalAgent(self.memory_manager)
        self.processing_agent = ProcessingAgent(config.MODEL_NAME)
        self.logger = logging.getLogger('master')

    def execute_query(self, query: str) -> str:
        """Execute the query by retrieving, processing, and directly responding."""
        try:
            relevant_memories = self.retrieval_agent.retrieve(query)
            self.logger.debug(f"Retrieved relevant memories: {relevant_memories}")

            context_documents = [(memory[0], memory[1], memory[2]) for memory in relevant_memories]

            response = self.processing_agent.process(query, context_documents)
            self.logger.debug(f"Generated response: {response}")

            self.memory_manager.save_memory(query, response)

            return response
        except Exception as e:
            self.logger.error(f"Error executing query '{query}': {str(e)}", exc_info=True)
            raise

    def get_recent_memories(self, limit: int) -> List[Tuple[str, str]]:
        """Retrieve recent memories from the memory manager."""
        return self.memory_manager.get_memories(limit)