import logging
import aiosqlite
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
import numpy as np
from config import config
import os
import asyncio

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, api_key: str):
        self.master_db_path = "/app/data/master.db"
        self.personal_db_path = self.get_personal_db_path()
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.container_id = self.get_container_id()
        asyncio.run(self.initialize_databases())

    def get_container_id(self):
        return os.environ.get('HOSTNAME', 'local')

    def get_personal_db_path(self):
        container_id = self.get_container_id()
        return f"/app/data/{container_id}.db"

    async def initialize_databases(self):
        os.makedirs("/app/data", exist_ok=True)
        await self.create_tables(self.master_db_path)
        await self.create_tables(self.personal_db_path)

    async def create_tables(self, db_path):
        async with aiosqlite.connect(db_path) as db:
            if db_path == self.master_db_path:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY,
                        query TEXT,
                        result TEXT,
                        embedding BLOB,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        author TEXT
                    )
                """)
            else:
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
        logger.debug(f"Ensured memories table exists in {db_path}")

    async def save_memory(self, query: str, result: str):
        try:
            embedding = np.array(await self.embeddings.aembed_query(query)).tobytes()
            
            # Save to personal database
            async with aiosqlite.connect(self.personal_db_path) as db:
                await db.execute("""
                    INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)
                """, (query, result, embedding))
                await db.commit()
            
            # Save to master database
            async with aiosqlite.connect(self.master_db_path) as db:
                await db.execute("""
                    INSERT INTO memories (query, result, embedding, author) VALUES (?, ?, ?, ?)
                """, (query, result, embedding, self.container_id))
                await db.commit()
            
            logger.debug(f"Saved memory for query: {query} with result: {result}")
        except Exception as e:
            logger.error(f"Error saving memory for query '{query}': {str(e)}", exc_info=True)
            raise

    async def retrieve_relevant_memories(self, query: str, threshold: float = config.MEMORY_RETRIEVAL_THRESHOLD) -> List[Tuple[str, str, str]]:
        try:
            query_embedding = np.array(await self.embeddings.aembed_query(query))
            
            # First, retrieve from personal database
            personal_memories = await self._retrieve_from_db(self.personal_db_path, query_embedding, threshold)
            
            # If we don't have enough personal memories, retrieve from master database
            if len(personal_memories) < 3:
                master_memories = await self._retrieve_from_db(self.master_db_path, query_embedding, threshold, exclude_author=self.container_id)
                combined_memories = personal_memories + master_memories
                combined_memories.sort(key=lambda x: x[2], reverse=True)
                return combined_memories[:config.MEMORY_RETRIEVAL_LIMIT]
            
            return personal_memories[:config.MEMORY_RETRIEVAL_LIMIT]
        except Exception as e:
            logger.error(f"Error retrieving relevant memories for query '{query}': {str(e)}", exc_info=True)
            raise

    async def _retrieve_from_db(self, db_path: str, query_embedding: np.ndarray, threshold: float, exclude_author: str = None) -> List[Tuple[str, str, float, str]]:
        async with aiosqlite.connect(db_path) as db:
            if exclude_author:
                async with db.execute("SELECT query, result, embedding, timestamp FROM memories WHERE author != ?", (exclude_author,)) as cursor:
                    memories = await cursor.fetchall()
            else:
                async with db.execute("SELECT query, result, embedding, timestamp FROM memories") as cursor:
                    memories = await cursor.fetchall()

        relevant_memories = []
        for memory in memories:
            memory_query, memory_result, memory_embedding, timestamp = memory
            memory_embedding = np.frombuffer(memory_embedding)
            similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
            if similarity >= threshold:
                relevant_memories.append((memory_query, memory_result, similarity, timestamp))

        relevant_memories.sort(key=lambda x: x[2], reverse=True)
        return relevant_memories

    async def get_memories(self, limit: int = config.MEMORY_RETRIEVAL_LIMIT) -> List[Tuple[str, str]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("""
                SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
            """, (limit,)) as cursor:
                memories = await cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories