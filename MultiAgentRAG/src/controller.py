# src/controller.py

import logging
from typing import List, Tuple
from langchain.schema import Document
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.processing_agent import ProcessingAgent
from src.agents.response_agent import ResponseAgent
from src.memory.memory_manager import MemoryManager
from config import Config

class Controller:
    def __init__(self):
        self.memory_manager = MemoryManager(Config.MEMORY_DB_PATH, Config.OPENAI_API_KEY)
        self.retrieval_agent = RetrievalAgent(self.memory_manager)
        self.processing_agent = ProcessingAgent(Config.MODEL_NAME)
        self.response_agent = ResponseAgent(Config.MODEL_NAME)
        self.logger = logging.getLogger('master')

    def execute_query(self, query: str) -> str:
        """Execute the query by retrieving, processing, and generating a response."""
        try:
            relevant_memories = self.retrieval_agent.retrieve(query)
            self.logger.debug(f"Retrieved relevant memories: {relevant_memories}")

            context_documents = [Document(page_content=f"{memory[0]}\n{memory[1]}") for memory in relevant_memories]

            result = self.processing_agent.process(query, context_documents)
            self.logger.debug(f"Processing result: {result}")

            response = self.response_agent.generate_response(query, result)
            self.logger.debug(f"Generated response: {response}")

            self.memory_manager.save_memory(query, response)

            return response
        except Exception as e:
            self.logger.error(f"Error executing query '{query}': {str(e)}", exc_info=True)
            raise

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        """Retrieve a list of memories up to the specified limit."""
        try:
            memories = self.memory_manager.get_memories(limit)
            self.logger.debug(f"Memories retrieved: {memories}")
            return memories
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {str(e)}", exc_info=True)
            raise
