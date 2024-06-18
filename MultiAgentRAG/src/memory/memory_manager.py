# src/memory/memory_manager.py

import logging
import sqlite3
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
import numpy as np
import os

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                query TEXT,
                result TEXT,
                embedding BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.debug("Ensured memories table exists")

    def save_memory(self, query: str, result: str):
        embedding = np.array(self.embeddings.embed_query(query)).tobytes()  # Ensure embedding is a numpy array
        self.conn.execute("""
            INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)
        """, (query, result, embedding))
        self.conn.commit()
        logger.debug(f"Saved memory for query: {query} with result: {result}")

    def load_precomputed_embeddings(self):
        # Load precomputed embeddings from the database if needed
        pass

    def get_all_memories(self) -> List[Tuple[str, str, bytes]]:
        cursor = self.conn.execute("SELECT query, result, embedding FROM memories")
        memories = cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        cursor = self.conn.execute("""
            SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        memories = cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    def retrieve_relevant_memories(self, query: str, threshold: float = 0.75) -> List[Tuple[str, str]]:
        query_embedding = np.array(self.embeddings.embed_query(query))  # Ensure embedding is a numpy array
        all_memories = self.get_all_memories()
        relevant_memories = []

        for memory in all_memories:
            memory_query, memory_result, memory_embedding = memory
            memory_embedding = np.frombuffer(memory_embedding)  # Convert back from bytes to numpy array
            similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
            if similarity >= threshold:
                relevant_memories.append((memory_query, memory_result, similarity))

        relevant_memories.sort(key=lambda x: x[2], reverse=True)
        return [(memory[0], memory[1]) for memory in relevant_memories[:3]]  # Return top 3 relevant memories

    def reset_database(self):
        self.conn.close()
        os.remove(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()
        logger.debug("Database reset and tables recreated")
