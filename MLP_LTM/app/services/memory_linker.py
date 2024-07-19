from datetime import datetime
from app.db.memory_database import MemoryDatabase, Memory
from app.services.embedding_service import EmbeddingService
from app.config import config
from app.utils.logging import get_logger
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger('custom')

class MemoryLinker:
    def __init__(self, memory_db: MemoryDatabase, embedding_service: EmbeddingService):
        self.memory_db = memory_db
        self.embedding_service = embedding_service
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.memory_contents = []

    def update_tfidf_matrix(self):
        try:
            with self.memory_db.get_db() as db:
                memories = db.query(Memory).all()
                if memories:
                    self.memory_contents = [m.query if config.MEMORY_LINKING['query_only_linking'] else f"{m.query} {m.response}" for m in memories]
                    if self.memory_contents:
                        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.memory_contents)
                        logger.info(f"TF-IDF matrix updated with {len(self.memory_contents)} memories")
                    else:
                        logger.info("No valid memory contents found. TF-IDF matrix not updated.")
                else:
                    logger.info("No memories in the database. TF-IDF matrix not updated.")
                    self.memory_contents = []
                    self.tfidf_matrix = None
        except Exception as e:
            logger.error(f"Error updating TF-IDF matrix: {str(e)}", exc_info=True)
            self.memory_contents = []
            self.tfidf_matrix = None

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

    def _get_embedding_similarity(self, memory1: Memory, memory2: Memory) -> float:
        query_similarity = self.embedding_service.cosine_similarity(
            np.fromstring(memory1.query_embedding, sep=','),
            np.fromstring(memory2.query_embedding, sep=',')
        )
        if config.MEMORY_LINKING['query_only_linking']:
            return query_similarity
        else:
            response_similarity = 0
            if memory1.response_embedding and memory2.response_embedding:
                response_similarity = self.embedding_service.cosine_similarity(
                    np.fromstring(memory1.response_embedding, sep=','),
                    np.fromstring(memory2.response_embedding, sep=',')
                )
            return max(query_similarity, response_similarity)

    def _should_link(self, memory1: Memory, memory2: Memory) -> bool:
        embedding_similarity = self._get_embedding_similarity(memory1, memory2)
        logger.debug(f"Embedding similarity between memory {memory1.id} and {memory2.id} is {embedding_similarity}")

        keyword_similarity = 0
        if config.MEMORY_LINKING['keyword_matching']['enabled']:
            content1 = memory1.query if config.MEMORY_LINKING['query_only_linking'] else f"{memory1.query} {memory1.response}"
            content2 = memory2.query if config.MEMORY_LINKING['query_only_linking'] else f"{memory2.query} {memory2.response}"
            keyword_similarity = self._keyword_similarity(content1, content2) or 0
            logger.debug(f"Keyword similarity between memory {memory1.id} and {memory2.id} is {keyword_similarity}")

        combined_score = (
            config.MEMORY_LINKING['keyword_matching']['embedding_weight'] * embedding_similarity +
            config.MEMORY_LINKING['keyword_matching']['keyword_weight'] * keyword_similarity
        )
        logger.info(f"Combined similarity score between memory {memory1.id} and {memory2.id} is {combined_score}")

        if combined_score > config.MEMORY_LINKING['similarity_threshold']:
            logger.info(f"Memories {memory1.id} and {memory2.id} linked with combined score: {combined_score}")
            return True
        else:
            logger.debug(f"Memories {memory1.id} and {memory2.id} not linked: combined score {combined_score} below threshold {config.MEMORY_LINKING['similarity_threshold']}")
            return False

    def _keyword_similarity(self, content1: str, content2: str) -> float:
        if self.tfidf_matrix is None or len(self.memory_contents) == 0:
            return 0
        new_vector = self.tfidf_vectorizer.transform([content1, content2])
        similarity = cosine_similarity(new_vector[0:1], new_vector[1:2])[0][0]
        return similarity

    def update_links_for_query(self, query_embedding: np.ndarray, relevant_memories: List[Memory]):
        logger.debug("Updating links for query-based retrieval")
        with self.memory_db.get_db() as db:
            for i, memory1 in enumerate(relevant_memories):
                for memory2 in relevant_memories[i+1:]:
                    if self._should_link(memory1, memory2):
                        try:
                            self.memory_db.add_link(memory1.id, memory2.id)
                            logger.debug(f"Created link between memory ID {memory1.id} and memory ID {memory2.id}")
                        except ValueError as e:
                            logger.warning(f"Could not create link: {str(e)}")

        logger.info("Completed updating links for query-based retrieval")