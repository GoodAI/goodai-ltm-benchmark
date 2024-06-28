import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import aiosqlite
from typing import List, Tuple, Dict
from src.memory.memory_manager import MemoryManager
from datetime import datetime

class EnhancedMemoryManager(MemoryManager):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.tfidf_vectorizer = None
        self.bm25 = None
        self.corpus = []

    async def initialize(self):
        await super().initialize()
        await self._load_corpus()
        if self.corpus:
            self._update_indexing()

    async def _load_corpus(self):
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT query, result FROM memories") as cursor:
                memories = await cursor.fetchall()
        self.corpus = [f"{query} {result}" for query, result in memories]

    def _update_indexing(self):
        if self.corpus:
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(self.corpus)
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

    async def save_memory(self, query: str, result: str):
        await super().save_memory(query, result)
        self.corpus.append(f"{query} {result}")
        self._update_indexing()

    async def retrieve_relevant_memories(self, query: str, threshold: float = 0.75, return_metadata: bool = False) -> List[Tuple[str, str, str]]:
        if not self.corpus:
            return []

        query_embedding = np.array(await self.embeddings.aembed_query(query))
        
        l2_memories = await self._retrieve_l2_norm(query_embedding, threshold)
        cosine_memories = await self._retrieve_cosine_similarity(query_embedding, threshold)
        bm25_memories = await self._retrieve_bm25(query, threshold)
        jaccard_memories = await self._retrieve_jaccard_similarity(query, threshold)

        combined_memories = self._combine_unique_memories([l2_memories, cosine_memories, bm25_memories, jaccard_memories])

        if return_metadata:
            return {
                "memories": [(memory[0], memory[1], memory[3]) for memory in combined_memories],
                "metadata": {
                    "l2_count": len(l2_memories),
                    "cosine_count": len(cosine_memories),
                    "bm25_count": len(bm25_memories),
                    "jaccard_count": len(jaccard_memories)
                }
            }
        else:
            return [(memory[0], memory[1], memory[3]) for memory in combined_memories]

    async def _retrieve_l2_norm(self, query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, float, str]]:
        return await super()._retrieve_from_db(self.personal_db_path, query_embedding, threshold)

    async def _retrieve_cosine_similarity(self, query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, float, str]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT query, result, embedding, timestamp FROM memories") as cursor:
                memories = await cursor.fetchall()

        relevant_memories = []
        for memory in memories:
            memory_query, memory_result, memory_embedding, timestamp = memory
            memory_embedding = np.frombuffer(memory_embedding)
            similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
            if similarity >= threshold:
                relevant_memories.append((memory_query, memory_result, similarity, timestamp))

        return relevant_memories

    async def _retrieve_bm25(self, query: str, threshold: float) -> List[Tuple[str, str, float, str]]:
        if not self.bm25:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        if len(scores) == 0:
            return []
        
        max_score = np.max(scores)
        if max_score == 0:
            return []
        
        normalized_scores = scores / max_score  # Normalize scores to [0, 1] range

        relevant_memories = []
        async with aiosqlite.connect(self.personal_db_path) as db:
            for i, score in enumerate(normalized_scores):
                if score >= threshold:
                    async with db.execute("SELECT query, result, timestamp FROM memories WHERE rowid = ?", (i+1,)) as cursor:
                        memory = await cursor.fetchone()
                        if memory:
                            relevant_memories.append((*memory, score))

        return relevant_memories

    async def _retrieve_jaccard_similarity(self, query: str, threshold: float) -> List[Tuple[str, str, float, str]]:
        query_set = set(query.lower().split())

        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT query, result, timestamp FROM memories") as cursor:
                memories = await cursor.fetchall()

        relevant_memories = []
        for memory in memories:
            memory_query, memory_result, timestamp = memory
            memory_set = set((memory_query + " " + memory_result).lower().split())
            similarity = len(query_set.intersection(memory_set)) / len(query_set.union(memory_set))
            if similarity >= threshold:
                relevant_memories.append((*memory, similarity))

        return relevant_memories

    def _combine_unique_memories(self, memory_lists: List[List[Tuple[str, str, float, str]]]) -> List[Tuple[str, str, str]]:
        seen = set()
        unique_memories = []
        for memories in memory_lists:
            for memory in memories:
                if memory[0] not in seen:  # Assuming the first element is a unique identifier
                    seen.add(memory[0])
                    unique_memories.append(memory)
        
        # Sort by similarity score (descending) and then by timestamp (descending)
        return sorted(
            unique_memories,
            key=lambda x: (-float(x[2]) if isinstance(x[2], (int, float)) else 0,
                           datetime.fromisoformat(x[3]) if isinstance(x[3], str) else datetime.min),
            reverse=True
        )