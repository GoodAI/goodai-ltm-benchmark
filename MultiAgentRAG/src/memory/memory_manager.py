import logging
import aiosqlite
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
import numpy as np
from config import config

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    async def create_tables(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    query TEXT,
                    result TEXT,
                    embedding BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()
        logger.debug("Ensured memories table exists")

    async def save_memory(self, query: str, result: str):
        try:
            embedding = np.array(await self.embeddings.aembed_query(query)).tobytes()
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)
                """, (query, result, embedding))
                await db.commit()
            logger.debug(f"Saved memory for query: {query} with result: {result}")
        except Exception as e:
            logger.error(f"Error saving memory for query '{query}': {str(e)}", exc_info=True)
            raise

    async def retrieve_relevant_memories(self, query: str, threshold: float = config.MEMORY_RETRIEVAL_THRESHOLD) -> List[Tuple[str, str, str]]:
        try:
            query_embedding = np.array(await self.embeddings.aembed_query(query))
            all_memories = await self.get_all_memories()
            relevant_memories = []

            for memory in all_memories:
                memory_query, memory_result, memory_embedding, timestamp = memory
                memory_embedding = np.frombuffer(memory_embedding)
                similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
                if similarity >= threshold:
                    relevant_memories.append((memory_query, memory_result, similarity, timestamp))

            if not relevant_memories:
                return []

            relevant_memories.sort(key=lambda x: (x[2], x[3]), reverse=True)
            return [(memory[0], memory[1], memory[3]) for memory in relevant_memories[:config.MEMORY_RETRIEVAL_LIMIT]]
        except Exception as e:
            logger.error(f"Error retrieving relevant memories for query '{query}': {str(e)}", exc_info=True)
            raise

    async def get_all_memories(self) -> List[Tuple[str, str, bytes, str]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT query, result, embedding, timestamp FROM memories") as cursor:
                memories = await cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    async def get_memories(self, limit: int = config.MEMORY_RETRIEVAL_LIMIT) -> List[Tuple[str, str]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
            """, (limit,)) as cursor:
                memories = await cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories