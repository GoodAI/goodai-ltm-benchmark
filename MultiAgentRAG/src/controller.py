# src/controller.py

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

    def execute_query(self, query: str) -> str:
        # Retrieve relevant documents
        context_documents = self.retrieval_agent.retrieve(query)
        print(f"Retrieved documents: {context_documents}")  # Debugging statement
        print(f"Number of documents retrieved: {len(context_documents)}")  # Debugging statement
        
        if not context_documents:
            raise ValueError("No documents retrieved")

        # Process query with context
        result = self.processing_agent.process(query, context_documents)
        print(f"Processing result: {result}")  # Debugging statement

        # Generate final response
        response = self.response_agent.generate_response(query, result)
        print(f"Generated response: {response}")  # Debugging statement

        # Save memory
        self.memory_manager.save_memory(query, response)

        return response

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        memories = self.memory_manager.get_memories(limit)
        print(f"Memories retrieved: {memories}")  # Debugging statement
        return memories
