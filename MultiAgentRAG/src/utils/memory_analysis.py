from collections import Counter
import numpy as np
import aiosqlite

class MemoryAnalyzer:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def analyze_distribution(self):
        async with aiosqlite.connect(self.memory_manager.master_db_path) as db:
            async with db.execute("SELECT author, COUNT(*) FROM memories GROUP BY author") as cursor:
                distribution = await cursor.fetchall()
        return dict(distribution)

    async def find_most_accessed(self, limit=10):
        # Assuming we've added an 'access_count' column to the memories table
        async with aiosqlite.connect(self.memory_manager.master_db_path) as db:
            async with db.execute("SELECT id, query, access_count FROM memories ORDER BY access_count DESC LIMIT ?", (limit,)) as cursor:
                most_accessed = await cursor.fetchall()
        return most_accessed

    async def find_most_similar(self, query, limit=10):
        query_embedding = await self.memory_manager.embeddings.aembed_query(query)
        all_memories = await self.memory_manager.get_all_memories()
        
        similarities = []
        for memory in all_memories:
            memory_embedding = np.frombuffer(memory[2])
            similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
            similarities.append((memory[0], memory[1], similarity))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:limit]