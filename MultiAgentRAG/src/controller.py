# src/controller.py

import logging
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from agents.retrieval_agent import RetrievalAgent
from agents.processing_agent import ProcessingAgent
from agents.response_agent import ResponseAgent
from memory.memory_manager import MemoryManager

class Controller:
    def __init__(self, vectorstore: FAISS, model_name: str, memory_db_path: str):
        self.retrieval_agent = RetrievalAgent(vectorstore)
        self.processing_agent = ProcessingAgent(model_name)
        self.response_agent = ResponseAgent(model_name)
        self.memory_manager = MemoryManager(memory_db_path)
        self.logger = logging.getLogger('master')

    def execute_query(self, query: str) -> str:
        # Retrieve relevant documents
        context_documents = self.retrieval_agent.retrieve(query)
        self.logger.debug(f"Retrieved documents: {context_documents}")
        self.logger.debug(f"Number of documents retrieved: {len(context_documents)}")
        
        if not context_documents:
            raise ValueError("No documents retrieved")

        # Process query with context
        result = self.processing_agent.process(query, context_documents)
        self.logger.debug(f"Processing result: {result}")

        # Generate final response
        response = self.response_agent.generate_response(query, result)
        self.logger.debug(f"Generated response: {response}")

        # Save memory
        self.memory_manager.save_memory(query, response)

        return response

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        memories = self.memory_manager.get_memories(limit)
        self.logger.debug(f"Memories retrieved: {memories}")
        return memories
