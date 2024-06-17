import logging
import sqlite3
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
import numpy as np
import glob
import json

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        self.embeddings = OpenAIEmbeddings()
        self.load_memories()

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

    def load_memories(self):
        json_files = glob.glob("json_output/*.json")
        
        for file_path in json_files:
            with open(file_path, 'r') as json_file:
                memory_data = json.load(json_file)
                query = memory_data['query']
                result = memory_data['result']
                self.save_memory(query, result)
        
        logger.debug(f"Loaded {len(json_files)} memories from JSON files")

    def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]: #? is this used? 
        cursor = self.conn.execute("""
            SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        memories = cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    def retrieve_relevant_memories(self, query: str, threshold: float = 0.75) -> List[Tuple[str, str]]:
        #! This is as of yet not scalable as the entire database is returned. 
        # To address this we will need to sort the sematic title search. Potentially use this as a hierarchy structure or temporal structure instead. 
        cursor = self.conn.execute("SELECT query, result FROM memories")
        all_memories = cursor.fetchall()
        
        query_embedding = self.embeddings.embed_query(query)
        relevant_memories = []
        memory_texts = set()
        
        for memory in all_memories:
            memory_query = memory[0]
            memory_result = memory[1]
            memory_text = f"{memory_query}\n{memory_result}"
            if memory_text in memory_texts:
                continue
            
            memory_embedding = self.embeddings.embed_query(memory_query)
            similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
            
            if similarity >= threshold:
                relevant_memories.append((memory, similarity))
                memory_texts.add(memory_text)
        
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory[0] for memory in relevant_memories]
