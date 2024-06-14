# src/memory/memory_manager.py

import logging
import sqlite3
from typing import List, Tuple

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        logger.debug(f"Connected to SQLite database at {db_path}")

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                query TEXT,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.debug("Ensured memories table exists")

    def save_memory(self, query: str, result: str):
        self.conn.execute("""
            INSERT INTO memories (query, result) VALUES (?, ?)
        """, (query, result))
        self.conn.commit()
        logger.debug(f"Saved memory for query: {query} with result: {result}")

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        cursor = self.conn.execute("""
            SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        memories = cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories
