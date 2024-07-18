# app/db/memory_manager.py
from app.db.memory_database import MemoryDatabase
from app.services.embedding_service import EmbeddingService
from app.services.memory_linker import MemoryLinker
from app.services.memory_retriever import MemoryRetriever
from app.config import config
from app.utils.logging import get_logger
from typing import List

logger = get_logger('custom')

class MemoryManager:
    def __init__(self, db_url: str):
        self.memory_db = MemoryDatabase(db_url)
        self.embedding_service = EmbeddingService()
        self.memory_linker = MemoryLinker(self.memory_db, self.embedding_service)
        self.memory_retriever = MemoryRetriever(self.memory_db, self.embedding_service)

    def initialize(self):
        try:
            self.memory_db.initialize()
            logger.info("MemoryManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MemoryManager: {str(e)}", exc_info=True)
            raise

    async def create_memory_with_query(self, query: str) -> int:
        try:
            logger.debug(f"Creating new memory with query: {query[:50]}...")
            query_embedding = await self.embedding_service.get_embedding(query)
            memory_id = self.memory_db.create_memory(query, query_embedding)
            logger.info(f"Memory created with query: ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error creating memory with query: {str(e)}", exc_info=True)
            raise

    async def update_memory_with_response(self, memory_id: int, response: str):
        try:
            logger.debug(f"Updating memory ID {memory_id} with response: {response[:50]}...")
            response_embedding = await self.embedding_service.get_embedding(response)
            self.memory_db.update_memory_response(memory_id, response, response_embedding)
            
            if config.MEMORY_LINKING['enabled']:
                memory = self.memory_db.get_memory(memory_id)
                self.memory_linker.update_links(memory)
            
            logger.info(f"Memory ID {memory_id} updated with response")
        except Exception as e:
            logger.error(f"Error updating memory with response: {str(e)}", exc_info=True)
            raise

    async def get_relevant_memories(self, query: str, current_memory_id: int, top_k: int = None) -> List[str]:
        try:
            return await self.memory_retriever.get_relevant_memories(query, current_memory_id, top_k or config.RETRIEVAL['top_k'])
        except Exception as e:
            logger.error(f"Error getting relevant memories: {str(e)}", exc_info=True)
            raise