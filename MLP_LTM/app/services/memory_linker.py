# app/services/memory_linker.py
from app.db.memory_database import MemoryDatabase, Memory
from app.services.embedding_service import EmbeddingService
from app.config import config
from app.utils.logging import get_logger
from typing import List
import numpy as np

logger = get_logger('custom')

class MemoryLinker:
    def __init__(self, memory_db: MemoryDatabase, embedding_service: EmbeddingService):
        self.memory_db = memory_db
        self.embedding_service = embedding_service

    def update_links(self, memory: Memory):
        try:
            logger.debug(f"Updating links for memory: ID {memory.id}")
            all_memories = self.memory_db.get_all_memories()
            links_count = 0
            for other_memory in all_memories:
                if other_memory.id != memory.id and self._should_link(memory, other_memory):
                    self.memory_db.add_link(memory.id, other_memory.id)
                    links_count += 1
                    logger.debug(f"Linked memory ID {other_memory.id} to memory ID {memory.id}")
                
                if config.MEMORY_LINKING['max_links_per_memory'] is not None and links_count >= config.MEMORY_LINKING['max_links_per_memory']:
                    break
            logger.info(f"Updated links for memory ID {memory.id}. Total links: {links_count}")
        except Exception as e:
            logger.error(f"Error updating links: {str(e)}", exc_info=True)
            raise

    def _should_link(self, memory1: Memory, memory2: Memory) -> bool:
        if config.MEMORY_LINKING['query_only_linking']:
            similarity = self.embedding_service.cosine_similarity(
                np.fromstring(memory1.query_embedding, sep=','),
                np.fromstring(memory2.query_embedding, sep=',')
            )
        else:
            query_similarity = self.embedding_service.cosine_similarity(
                np.fromstring(memory1.query_embedding, sep=','),
                np.fromstring(memory2.query_embedding, sep=',')
            )
            response_similarity = 0
            if memory1.response_embedding and memory2.response_embedding:
                response_similarity = self.embedding_service.cosine_similarity(
                    np.fromstring(memory1.response_embedding, sep=','),
                    np.fromstring(memory2.response_embedding, sep=',')
                )
            similarity = max(query_similarity, response_similarity)
        
        if similarity > config.MEMORY_LINKING['similarity_threshold']:
            return True
        
        if config.MEMORY_LINKING['keyword_matching']['enabled']:
            keyword_similarity = self._keyword_similarity(memory1.query + memory1.response, memory2.query + memory2.response)
            if keyword_similarity > config.MEMORY_LINKING['keyword_matching']['threshold']:
                return True
        
        return False

    def _keyword_similarity(self, content1: str, content2: str) -> float:
        # This is a placeholder function for keyword matching
        # Implement your keyword matching algorithm here
        # Return a similarity score between 0 and 1
        return 0.0

    def update_links_for_query(self, query_embedding: np.ndarray, relevant_memories: List[Memory]):
        logger.debug("Updating links for query-based retrieval")
        for memory in relevant_memories:
            similarity = self.embedding_service.cosine_similarity(
                query_embedding,
                np.fromstring(memory.query_embedding, sep=',')
            )
            if similarity > config.MEMORY_LINKING['similarity_threshold']:
                # Here you would implement the logic to create links based on the query
                # This might involve creating a temporary memory object for the query
                # or updating the links of the retrieved memories
                pass