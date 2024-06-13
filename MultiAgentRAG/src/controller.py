from typing import List
from langchain.schema import Document

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

        # Process query with context
        result = self.processing_agent.process(query, context_documents)

        # Generate final response
        response = self.response_agent.generate_response(query, result)

        # Save memory
        self.memory_manager.save_memory(query, response)

        return response

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        return self.memory_manager.get_memories(limit)