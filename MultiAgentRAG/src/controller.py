# src/controller.py

import logging
from typing import List, Tuple
from langchain.schema import Document
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.processing_agent import ProcessingAgent
from src.agents.response_agent import ResponseAgent
from src.memory.memory_manager import MemoryManager

class Controller:
    def __init__(self, model_name: str, memory_db_path: str, api_key: str):
        self.memory_manager = MemoryManager(memory_db_path, api_key)
        self.retrieval_agent = RetrievalAgent(self.memory_manager)
        self.processing_agent = ProcessingAgent(model_name)
        self.response_agent = ResponseAgent(model_name)
        self.logger = logging.getLogger('master')

    def execute_query(self, query: str) -> str:
        relevant_memories = self.retrieval_agent.retrieve(query)
        self.logger.debug(f"Retrieved relevant memories: {relevant_memories}")
        
        context_documents = [Document(page_content=f"{memory[0]}\n{memory[1]}") for memory in relevant_memories]
        
        result = self.processing_agent.process(query, context_documents)
        self.logger.debug(f"Processing result: {result}")

        response = self.response_agent.generate_response(query, result)
        self.logger.debug(f"Generated response: {response}")

        self.memory_manager.save_memory(query, response)

        return response

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        memories = self.memory_manager.get_memories(limit)
        self.logger.debug(f"Memories retrieved: {memories}")
        return memories
