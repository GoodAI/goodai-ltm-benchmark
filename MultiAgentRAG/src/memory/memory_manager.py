# src/memory/memory_manager.py

import logging
import sqlite3
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
import numpy as np
import os
from config import config
from datetime import datetime

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.create_tables()

    def create_tables(self):
        """Create the necessary tables if they don't already exist."""
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
        """Save a memory to the database."""
        try:
            embedding = np.array(self.embeddings.embed_query(query)).tobytes()
            self.conn.execute("""
                INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)
            """, (query, result, embedding))
            self.conn.commit()
            logger.debug(f"Saved memory for query: {query} with result: {result}")
        except Exception as e:
            logger.error(f"Error saving memory for query '{query}': {str(e)}", exc_info=True)
            raise

# src/memory/memory_manager.py



# Modify the return statement to include the timestamp
    def retrieve_relevant_memories(self, query: str, threshold: float = config.MEMORY_RETRIEVAL_THRESHOLD) -> List[Tuple[str, str, str]]:
        """Retrieve memories relevant to the query based on a similarity threshold, sorted by timestamp."""
        try:
            query_embedding = np.array(self.embeddings.embed_query(query))
            all_memories = self.get_all_memories()
            relevant_memories = []

            for memory in all_memories:
                memory_query, memory_result, memory_embedding, timestamp = memory
                memory_embedding = np.frombuffer(memory_embedding)
                similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
                if similarity >= threshold:
                    relevant_memories.append((memory_query, memory_result, similarity, timestamp))

            if not relevant_memories:
                return []

            # Sort by similarity in descending order
            relevant_memories.sort(key=lambda x: x[2], reverse=True)

            # Filter memories above the threshold and sort by timestamp in descending order
            filtered_memories = [mem for mem in relevant_memories if mem[2] >= threshold]
            filtered_memories.sort(key=lambda x: x[3], reverse=True)

            # Return the relevant memories with the timestamp
            return [(memory[0], memory[1], memory[3]) for memory in filtered_memories]
        except Exception as e:
            logger.error(f"Error retrieving relevant memories for query '{query}': {str(e)}", exc_info=True)
            raise


    def reset_database(self):
        self.conn.close()
        os.remove(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()
        logger.debug("Database reset and tables recreated")

    def load_precomputed_embeddings(self):
        # Load precomputed embeddings from the database if needed
        pass

    def get_all_memories(self) -> List[Tuple[str, str, bytes, str]]:
        cursor = self.conn.execute("SELECT query, result, embedding, timestamp FROM memories")
        memories = cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    def get_memories(self, limit: int = config.MEMORY_RETRIEVAL_LIMIT) -> List[Tuple[str, str]]:
        cursor = self.conn.execute("""
            SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        memories = cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories
