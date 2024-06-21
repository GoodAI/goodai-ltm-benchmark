# src/controller.py

# src/controller.py

import logging
from typing import List, Tuple
from langchain.schema import Document
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.processing_agent import ProcessingAgent
# Comment out the ResponseAgent import to avoid any unintentional use
# from src.agents.response_agent import ResponseAgent
from src.memory.memory_manager import MemoryManager
from config import Config

class Controller:
    def __init__(self):
        self.memory_manager = MemoryManager(Config.MEMORY_DB_PATH, Config.OPENAI_API_KEY)
        self.retrieval_agent = RetrievalAgent(self.memory_manager)
        self.processing_agent = ProcessingAgent(Config.MODEL_NAME)
        self.logger = logging.getLogger('master')

    def execute_query(self, query: str) -> str:
        """Execute the query by retrieving, processing, and directly responding."""
        try:
            relevant_memories = self.retrieval_agent.retrieve(query)
            self.logger.debug(f"Retrieved relevant memories: {relevant_memories}")

            # Adjusted to handle the new tuple format including timestamp
            context_documents = [(memory[0], memory[1], memory[2]) for memory in relevant_memories]

            # Directly return the result from the ProcessingAgent
            result = self.processing_agent.process(query, context_documents)
            self.logger.debug(f"Processing result: {result}")

            # Bypass the ResponseAgent
            response = result
            self.logger.debug(f"Generated response: {response}")

            self.memory_manager.save_memory(query, response)

            return response
        except Exception as e:
            self.logger.error(f"Error executing query '{query}': {str(e)}", exc_info=True)
            raise
        