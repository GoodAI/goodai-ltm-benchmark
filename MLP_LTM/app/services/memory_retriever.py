# app/services/memory_retriever.py
from app.db.memory_database import MemoryDatabase, Memory
from app.services.embedding_service import EmbeddingService
from app.config import config
from app.utils.logging import get_logger
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime

logger = get_logger('custom')

class MemoryRetriever:
    def __init__(self, memory_db: MemoryDatabase, embedding_service: EmbeddingService):
        self.memory_db = memory_db
        self.embedding_service = embedding_service

    async def get_relevant_memories(self, query: str, current_memory_id: int, top_k: int = None) -> Tuple[List[str], List[Memory], Dict[int, str]]:
        try:
            logger.debug(f"Retrieving relevant memories for query: {query[:50]}...")
            query_embedding = await self.embedding_service.get_embedding(query)
            
            with self.memory_db.get_db() as db:
                memories = db.query(Memory).filter(Memory.id != current_memory_id).all()
                similarities = self._calculate_similarities(memories, query_embedding)
                sorted_memories = sorted(zip(memories, similarities), key=lambda x: x[1], reverse=True)
                
                result = [(m, sim) for m, sim in sorted_memories if sim >= config.RETRIEVAL['min_similarity']]
                
                retrieval_info = {m.id: 'raw retrieval' for m, _ in result}
                
                if config.MEMORY_LINKING['enabled']:
                    result, retrieval_info = self._include_linked_memories(db, result, retrieval_info)
                
                formatted_memories, memory_ids = self._format_memories(result)
                memory_objects = [m for m, _ in result]
                
                total_matching_memories = len(formatted_memories)
                
                if top_k is not None:
                    formatted_memories = formatted_memories[:top_k]
                    memory_objects = memory_objects[:top_k]
                    memory_ids = memory_ids[:top_k]
                    retrieval_info = {k: v for k, v in retrieval_info.items() if k in memory_ids}
                
                logger.info(f"Retrieved {len(formatted_memories)} out of {total_matching_memories} relevant unique memories")
                self._log_retrieval_output(formatted_memories, memory_ids, retrieval_info)
                
                return formatted_memories, memory_objects, retrieval_info
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

    def _include_linked_memories(self, db, result: List[Tuple[Memory, float]], retrieval_info: Dict[int, str]) -> Tuple[List[Tuple[Memory, float]], Dict[int, str]]:
        linked_memories = set()
        processed_ids = set()

        def process_links(memory):
            if memory.id in processed_ids:
                return
            processed_ids.add(memory.id)
            
            for linked_memory in db.query(Memory).filter(Memory.id.in_([link.id for link in memory.links])).all():
                if linked_memory.id not in retrieval_info:
                    linked_memories.add((linked_memory, 0))  # We don't calculate similarity for linked memories
                    retrieval_info[linked_memory.id] = 'via link'
                process_links(linked_memory)

        for memory, _ in result:
            process_links(memory)

        result.extend(linked_memories)
        return result, retrieval_info

    def _format_memories(self, memories: List[Tuple[Memory, float]]) -> Tuple[List[str], List[int]]:
        unique_memories = {}
        for memory, sim in memories:
            if memory.id not in unique_memories:
                unique_memories[memory.id] = memory
        
        sorted_unique_memories = sorted(unique_memories.values(), key=lambda m: m.timestamp, reverse=True)
        
        formatted_memories = []
        memory_ids = []
        for memory in sorted_unique_memories:
            try:
                timestamp = datetime.strptime(memory.timestamp, config.MEMORY_FORMATTING['timestamp_format'])
                formatted_timestamp = timestamp.strftime(config.MEMORY_FORMATTING['timestamp_format'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format for memory ID {memory.id}: {memory.timestamp}")
                formatted_timestamp = "Unknown Time"

            if config.MEMORY_LINKING['query_only_linking']:
                formatted_memory = f"{formatted_timestamp} {memory.query}"
            else:
                formatted_memory = f"{formatted_timestamp} {memory.query}:{memory.response}" if memory.response else f"{formatted_timestamp} {memory.query}:"
            formatted_memories.append(formatted_memory)
            memory_ids.append(memory.id)
        
        return formatted_memories, memory_ids
    
    def _log_retrieval_output(self, formatted_memories: List[str], memory_ids: List[int], retrieval_info: Dict[int, str]):
        logger.info("Retrieval output:")
        for memory, memory_id in zip(formatted_memories, memory_ids):
            parts = memory.split()
            timestamp = " ".join(parts[:2])  # Assuming timestamp is the first two space-separated parts
            content = " ".join(parts[2:])[:50]  # Take up to 50 characters of the content
            logger.info(f"{memory_id:<5}{timestamp} {content:<50}{retrieval_info.get(memory_id, 'unknown')}")


