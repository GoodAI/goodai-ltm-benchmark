# app/services/memory_retriever.py
from app.db.memory_database import MemoryDatabase, Memory
from app.services.embedding_service import EmbeddingService
from app.config import config
from app.utils.logging import get_logger
from typing import List, Tuple
import numpy as np
from datetime import datetime

logger = get_logger('custom')

class MemoryRetriever:
    def __init__(self, memory_db: MemoryDatabase, embedding_service: EmbeddingService):
        self.memory_db = memory_db
        self.embedding_service = embedding_service

    async def get_relevant_memories(self, query: str, current_memory_id: int, top_k: int = None) -> List[str]:
        try:
            logger.debug(f"Retrieving relevant memories for query: {query[:50]}...")
            query_embedding = await self.embedding_service.get_embedding(query)
            
            with self.memory_db.get_db() as db:
                # Exclude the current memory from the query
                memories = db.query(Memory).filter(Memory.id != current_memory_id).all()
                similarities = self._calculate_similarities(memories, query_embedding)
                sorted_memories = sorted(zip(memories, similarities), key=lambda x: x[1], reverse=True)
                
                result = [(m, sim) for m, sim in sorted_memories if sim >= config.RETRIEVAL['min_similarity']]
                
                if config.MEMORY_LINKING['enabled']:
                    result = self._include_linked_memories(db, result, query_embedding, current_memory_id)
                
                formatted_memories = self._format_memories(result)
                
                total_matching_memories = len(formatted_memories)
                
                if top_k is None:
                    top_k = config.RETRIEVAL['top_k']
                
                if top_k is not None:
                    formatted_memories = formatted_memories[:top_k]
                
                logger.info(f"Retrieved {len(formatted_memories)} out of {total_matching_memories} relevant unique memories")
                
                return formatted_memories
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {str(e)}", exc_info=True)
            raise

    def _calculate_similarities(self, memories: List[Memory], query_embedding: np.ndarray) -> List[float]:
        if config.MEMORY_LINKING['query_only_linking']:
            return [
                self.embedding_service.cosine_similarity(query_embedding, np.fromstring(m.query_embedding, sep=','))
                for m in memories
            ]
        else:
            return [
                max(
                    self.embedding_service.cosine_similarity(query_embedding, np.fromstring(m.query_embedding, sep=',')),
                    self.embedding_service.cosine_similarity(query_embedding, np.fromstring(m.response_embedding, sep=',')) if m.response_embedding else 0
                )
                for m in memories
            ]

    def _include_linked_memories(self, db, result: List[Tuple[Memory, float]], query_embedding: np.ndarray, current_memory_id: int) -> List[Tuple[Memory, float]]:
        linked_memories = set()
        for memory, _ in result:
            for linked_memory in db.query(Memory).filter(Memory.id.in_([link.id for link in memory.links]), Memory.id != current_memory_id).all():
                if linked_memory not in [m for m, _ in result]:
                    linked_sim = max(
                        self.embedding_service.cosine_similarity(query_embedding, np.fromstring(linked_memory.query_embedding, sep=',')),
                        self.embedding_service.cosine_similarity(query_embedding, np.fromstring(linked_memory.response_embedding, sep=',')) if linked_memory.response_embedding else 0
                    )
                    if linked_sim >= config.RETRIEVAL['min_similarity']:
                        linked_memories.add((linked_memory, linked_sim))
        
        result.extend(linked_memories)
        return result

    def _format_memories(self, memories: List[Tuple[Memory, float]]) -> List[str]:
        unique_memories = {}
        for memory, sim in memories:
            if memory.id not in unique_memories:
                unique_memories[memory.id] = memory
        
        sorted_unique_memories = sorted(unique_memories.values(), key=lambda m: m.timestamp, reverse=True)
        
        formatted_memories = []
        for memory in sorted_unique_memories:
            try:
                timestamp = datetime.strptime(memory.timestamp, config.MEMORY_FORMATTING['timestamp_format'])
                formatted_timestamp = timestamp.strftime(config.MEMORY_FORMATTING['timestamp_format'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format for memory ID {memory.id}: {memory.timestamp}")
                formatted_timestamp = "Unknown Time"

            formatted_memory = f"{formatted_timestamp} {memory.query}:{memory.response}" if memory.response else f"{formatted_timestamp} {memory.query}:"
            formatted_memories.append(formatted_memory)
        
        return formatted_memories