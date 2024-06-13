import sqlite3
from typing import List, Tuple

class MemoryManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                query TEXT,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def save_memory(self, query: str, result: str):
        self.conn.execute("""
            INSERT INTO memories (query, result) VALUES (?, ?)
        """, (query, result))
        self.conn.commit()

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        cursor = self.conn.execute("""
            SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        return cursor.fetchall()