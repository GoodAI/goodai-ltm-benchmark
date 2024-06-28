import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import aiosqlite
from typing import List, Tuple, Dict, Optional
from src.memory.memory_manager import MemoryManager
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from config import Config

logger = logging.getLogger(__name__)

class EnhancedMemoryManager(MemoryManager):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.config = Config()

    async def initialize(self):
        """Initialize the EnhancedMemoryManager by loading the corpus and updating indexing."""
        await super().initialize()
        await self._load_corpus()
        self._update_indexing()

    async def _load_corpus(self):
        start_time = time.time()
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT query, result FROM memories") as cursor:
                memories = await cursor.fetchall()
        self.corpus = [f"{query} {result}" for query, result in memories]
        logger.info(f"Loaded corpus with {len(self.corpus)} entries in {time.time() - start_time:.2f} seconds")

    def _update_indexing(self):
        if self.corpus:
            start_time = time.time()
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(self.corpus)
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Updated indexing with {len(self.corpus)} documents in {time.time() - start_time:.2f} seconds")
        else:
            logger.warning("Corpus is empty. Skipping indexing update.")

    async def retrieve_relevant_memories(self, query: str, threshold: float = 0.75, return_metadata: bool = False) -> str:
        if not self.corpus:
            logger.warning("Corpus is empty. No memories to retrieve.")
            return ""

        start_time = time.time()
        query_embedding = np.array(await self.embeddings.aembed_query(query))
        
        try:
            async with aiosqlite.connect(self.personal_db_path) as db:
                async with db.execute("SELECT rowid, query, result, embedding, timestamp FROM memories") as cursor:
                    all_memories = await cursor.fetchall()
        except aiosqlite.Error as e:
            logger.error(f"Database error: {e}")
            return ""

        logger.debug(f"Retrieved {len(all_memories)} memories from database")

        with ThreadPoolExecutor(max_workers=4) as executor:
            l2_future = executor.submit(self._calculate_l2_norm, all_memories, query_embedding, threshold)
            cosine_future = executor.submit(self._calculate_cosine_similarity, all_memories, query_embedding, threshold)
            bm25_future = executor.submit(self._calculate_bm25, all_memories, query, threshold)
            jaccard_future = executor.submit(self._calculate_jaccard_similarity, all_memories, query, threshold)

            l2_memories = l2_future.result()
            cosine_memories = cosine_future.result()
            bm25_memories = bm25_future.result()
            jaccard_memories = jaccard_future.result()

        logger.debug(f"L2 memories: {len(l2_memories)}, Cosine memories: {len(cosine_memories)}, BM25 memories: {len(bm25_memories)}, Jaccard memories: {len(jaccard_memories)}")

        all_memories = {
            'L2 norm': l2_memories,
            'Cosine Similarity': cosine_memories,
            'BM25': bm25_memories,
            'Jaccard Similarity': jaccard_memories
        }

        formatted_output = self._format_memories(all_memories)

        logger.info(f"Retrieved and processed memories in {time.time() - start_time:.2f} seconds")

        if return_metadata:
            metadata = {
                "l2_count": len(l2_memories),
                "cosine_count": len(cosine_memories),
                "bm25_count": len(bm25_memories),
                "jaccard_count": len(jaccard_memories),
                "total_memories": len(all_memories),
                "processing_time": time.time() - start_time
            }
            return formatted_output, metadata
        else:
            return formatted_output

    def _calculate_l2_norm(self, memories: List[Tuple], query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        """Calculate L2 norm similarity."""
        return [
            (str(memory[0]), memory[1], memory[2], np.linalg.norm(query_embedding - np.frombuffer(memory[3])), memory[4])
            for memory in memories
            if np.linalg.norm(query_embedding - np.frombuffer(memory[3])) <= threshold
        ]

    def _calculate_cosine_similarity(self, memories: List[Tuple], query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        """Calculate cosine similarity."""
        return [
            (str(memory[0]), memory[1], memory[2], 
             np.dot(query_embedding, np.frombuffer(memory[3])) / (np.linalg.norm(query_embedding) * np.linalg.norm(np.frombuffer(memory[3]))),
             memory[4])
            for memory in memories
            if np.dot(query_embedding, np.frombuffer(memory[3])) / (np.linalg.norm(query_embedding) * np.linalg.norm(np.frombuffer(memory[3]))) >= threshold
        ]

    def _calculate_bm25(self, memories: List[Tuple], query: str, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        """Calculate BM25 similarity."""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        max_score = np.max(scores) if scores.size > 0 else 1  # Use np.max instead of max
        normalized_scores = scores / max_score

        return [
            (str(memory[0]), memory[1], memory[2], score, memory[4])
            for memory, score in zip(memories, normalized_scores)
            if score >= threshold
        ]
    
    def _calculate_jaccard_similarity(self, memories: List[Tuple], query: str, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        """Calculate Jaccard similarity."""
        query_set = set(query.lower().split())
        return [
            (str(memory[0]), memory[1], memory[2],
             len(set(memory[1].lower().split()) & query_set) / len(set(memory[1].lower().split()) | query_set),
             memory[4])
            for memory in memories
            if len(set(memory[1].lower().split()) & query_set) / len(set(memory[1].lower().split()) | query_set) >= threshold
        ]
    
    async def save_memory(self, query: str, result: str):
        await super().save_memory(query, result)
        await self._load_corpus()
        self._update_indexing()

    def _format_memories(self, all_memories: Dict[str, List[Tuple[str, str, str, float, str]]]) -> str:
        formatted_output = []
        seen_memories = set()

        for metric, memories in all_memories.items():
            if not memories:
                continue

            formatted_output.append(f"Similar by {metric} (ordered by timestamp - ascending):")
            sorted_memories = sorted(memories, key=lambda x: datetime.fromisoformat(x[4]))

            for memory in sorted_memories:
                memory_id, query, result, score, timestamp = memory
                if memory_id not in seen_memories:
                    formatted_output.append(f"<{memory_id}>, <{query}>, <{result}>, <{timestamp}>, <{score:.2f}>")
                    seen_memories.add(memory_id)
                else:
                    formatted_output.append(f"<{memory_id}>, <{score:.2f}>")

        return " ".join(formatted_output)