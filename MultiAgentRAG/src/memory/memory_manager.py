import logging
import aiosqlite
from typing import List, Tuple, Union, Dict
from langchain_openai import OpenAIEmbeddings
import numpy as np
from config import config
import os
from src.utils.enhanced_logging import log_execution_time, DatabaseLogger
import json
import csv
import io

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, api_key: str):
        self.master_db_path = "/app/data/master.db"
        self.personal_db_path = self.get_personal_db_path()
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.container_id = self.get_container_id()
        self.db_logger = DatabaseLogger(logging.getLogger('database'))

    async def initialize(self):
        await self.initialize_databases()

    async def initialize_databases(self):
        os.makedirs("/app/data", exist_ok=True)
        await self.create_tables(self.master_db_path)
        await self.create_tables(self.personal_db_path)
        await self.create_changelog_table(self.master_db_path)
        await self.create_changelog_table(self.personal_db_path)

    @log_execution_time(logger)
    async def inspect_database(self, db_path):
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT * FROM memories") as cursor:
                columns = [description[0] for description in cursor.description]
                rows = await cursor.fetchall()
        return {"columns": columns, "rows": rows}

    @log_execution_time(logger)
    async def compare_databases(self):
        personal_db = await self.inspect_database(self.personal_db_path)
        master_db = await self.inspect_database(self.master_db_path)
        
        personal_ids = set(row[0] for row in personal_db['rows'])
        master_ids = set(row[0] for row in master_db['rows'])
        
        only_in_personal = personal_ids - master_ids
        only_in_master = master_ids - personal_ids
        in_both = personal_ids.intersection(master_ids)
        
        return {
            "only_in_personal": list(only_in_personal),
            "only_in_master": list(only_in_master),
            "in_both": list(in_both)
        }

    def get_container_id(self):
        return os.environ.get('HOSTNAME', 'local')

    def get_personal_db_path(self):
        container_id = self.get_container_id()
        return f"/app/data/{container_id}.db"

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

    async def create_changelog_table(self, db_path):
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS changelog (
                    id INTEGER PRIMARY KEY,
                    operation TEXT,
                    memory_id INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

    async def log_change(self, db_path, operation, memory_id):
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO changelog (operation, memory_id) VALUES (?, ?)
            """, (operation, memory_id))
            await db.commit()

    async def save_memory(self, query: str, result: str):
        try:
            embedding = np.array(await self.embeddings.aembed_query(query)).tobytes()
            
            async with aiosqlite.connect(self.personal_db_path) as db:
                self.db_logger.log_query("INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)")
                cursor = await db.execute("""
                    INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)
                """, (query, result, embedding))
                memory_id = cursor.lastrowid
                await db.commit()
            self.db_logger.log_access(memory_id)
            await self.log_change(self.personal_db_path, "INSERT", memory_id)
            
            async with aiosqlite.connect(self.master_db_path) as db:
                self.db_logger.log_query("INSERT INTO memories (query, result, embedding, author) VALUES (?, ?, ?, ?)")
                cursor = await db.execute("""
                    INSERT INTO memories (query, result, embedding, author) VALUES (?, ?, ?, ?)
                """, (query, result, embedding, self.container_id))
                memory_id = cursor.lastrowid
                await db.commit()
            self.db_logger.log_access(memory_id)
            await self.log_change(self.master_db_path, "INSERT", memory_id)
            
            logger.debug(f"Saved memory for query: {query} with result: {result}")
        except Exception as e:
            logger.error(f"Error saving memory for query '{query}': {str(e)}", exc_info=True)
            raise
        
    async def retrieve_relevant_memories(self, query: str, threshold: float = config.MEMORY_RETRIEVAL_THRESHOLD, return_metadata: bool = False) -> Union[List[Tuple[str, str, str]], Dict]:
        try:
            query_embedding = np.array(await self.embeddings.aembed_query(query))
            
            personal_memories = await self._retrieve_from_db(self.personal_db_path, query_embedding, threshold)
            self.db_logger.log_query("SELECT query, result, embedding, timestamp FROM memories")
            
            metadata = {"queried_databases": ["personal"]}
            
            if len(personal_memories) < 3:
                master_memories = await self._retrieve_from_db(self.master_db_path, query_embedding, threshold, exclude_author=self.container_id)
                combined_memories = personal_memories + master_memories
                combined_memories.sort(key=lambda x: x[2], reverse=True)
                relevant_memories = combined_memories[:config.MEMORY_RETRIEVAL_LIMIT]
                metadata["queried_databases"].append("master")
            else:
                relevant_memories = personal_memories[:config.MEMORY_RETRIEVAL_LIMIT]
            
            if return_metadata:
                metadata["similarity_scores"] = [memory[2] for memory in relevant_memories]
                return {
                    "memories": [(memory[0], memory[1], memory[3]) for memory in relevant_memories],
                    "metadata": metadata
                }
            else:
                return [(memory[0], memory[1], memory[3]) for memory in relevant_memories]
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
    
    async def export_memories(self, db_path, format='json'):
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT * FROM memories") as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]

        if format == 'json':
            data = [dict(zip(columns, row)) for row in rows]
            return json.dumps(data, default=str)
        elif format == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(rows)
            return output.getvalue()
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'.")
        
    async def import_memories(self, db_path, data, format='json'):
        if format == 'json':
            memories = json.loads(data)
        elif format == 'csv':
            reader = csv.DictReader(io.StringIO(data))
            memories = list(reader)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'.")

        async with aiosqlite.connect(db_path) as db:
            for memory in memories:
                await db.execute("""
                    INSERT INTO memories (query, result, embedding, timestamp, author)
                    VALUES (?, ?, ?, ?, ?)
                """, (memory['query'], memory['result'], memory['embedding'], memory['timestamp'], memory.get('author')))
            await db.commit()

    async def explain_memory_retrieval(self, query: str, threshold: float = config.MEMORY_RETRIEVAL_THRESHOLD):
        explanation = []
        query_embedding = np.array(await self.embeddings.aembed_query(query))
        
        explanation.append(f"1. Converted query '{query}' to embedding.")
        
        personal_memories = await self._retrieve_from_db(self.personal_db_path, query_embedding, threshold)
        explanation.append(f"2. Retrieved {len(personal_memories)} memories from personal database.")
        
        if len(personal_memories) < 3:
            explanation.append("3. Less than 3 relevant memories found in personal database. Searching master database.")
            master_memories = await self._retrieve_from_db(self.master_db_path, query_embedding, threshold, exclude_author=self.container_id)
            explanation.append(f"4. Retrieved {len(master_memories)} additional memories from master database.")
            combined_memories = personal_memories + master_memories
            combined_memories.sort(key=lambda x: x[2], reverse=True)
            relevant_memories = combined_memories[:config.MEMORY_RETRIEVAL_LIMIT]
        else:
            explanation.append("3. Sufficient memories found in personal database. Not searching master database.")
            relevant_memories = personal_memories[:config.MEMORY_RETRIEVAL_LIMIT]
        
        explanation.append(f"5. Selected top {len(relevant_memories)} memories based on similarity.")
        for i, memory in enumerate(relevant_memories, 1):
            explanation.append(f"   {i}. Memory ID: {memory[0]}, Similarity: {memory[2]:.4f}")
        
        return "\n".join(explanation)
    
    async def perform_consistency_check(self):
        discrepancies = []
        
        async with aiosqlite.connect(self.personal_db_path) as personal_db, \
                   aiosqlite.connect(self.master_db_path) as master_db:
            async with personal_db.execute("SELECT id, query, result FROM memories") as personal_cursor:
                personal_memories = await personal_cursor.fetchall()

            for p_id, p_query, p_result in personal_memories:
                async with master_db.execute("SELECT id, query, result FROM memories WHERE author = ? AND query = ?", 
                                             (self.container_id, p_query)) as master_cursor:
                    master_memory = await master_cursor.fetchone()

                if master_memory is None:
                    discrepancies.append({
                        "type": "missing_in_master",
                        "personal_id": p_id,
                        "query": p_query
                    })
                elif master_memory[2] != p_result:
                    discrepancies.append({
                        "type": "content_mismatch",
                        "personal_id": p_id,
                        "master_id": master_memory[0],
                        "query": p_query,
                        "personal_result": p_result,
                        "master_result": master_memory[2]
                    })

            # Check for memories in master that are not in personal
            async with master_db.execute("SELECT id, query FROM memories WHERE author = ?", 
                                         (self.container_id,)) as master_cursor:
                master_memories = await master_cursor.fetchall()

            personal_queries = set(memory[1] for memory in personal_memories)
            for m_id, m_query in master_memories:
                if m_query not in personal_queries:
                    discrepancies.append({
                        "type": "missing_in_personal",
                        "master_id": m_id,
                        "query": m_query
                    })

        return discrepancies

    async def fix_discrepancies(self, discrepancies):
        for discrepancy in discrepancies:
            if discrepancy["type"] == "missing_in_master":
                await self._copy_memory_to_master(discrepancy["personal_id"])
            elif discrepancy["type"] == "missing_in_personal":
                await self._copy_memory_to_personal(discrepancy["master_id"])
            elif discrepancy["type"] == "content_mismatch":
                await self._resolve_content_mismatch(discrepancy)

    async def _copy_memory_to_master(self, personal_id):
        async with aiosqlite.connect(self.personal_db_path) as personal_db, \
                   aiosqlite.connect(self.master_db_path) as master_db:
            async with personal_db.execute("SELECT query, result, embedding FROM memories WHERE id = ?", 
                                           (personal_id,)) as cursor:
                memory = await cursor.fetchone()
            
            if memory:
                await master_db.execute("""
                    INSERT INTO memories (query, result, embedding, author)
                    VALUES (?, ?, ?, ?)
                """, (*memory, self.container_id))
                await master_db.commit()

    async def _copy_memory_to_personal(self, master_id):
        async with aiosqlite.connect(self.master_db_path) as master_db, \
                   aiosqlite.connect(self.personal_db_path) as personal_db:
            async with master_db.execute("SELECT query, result, embedding FROM memories WHERE id = ?", 
                                         (master_id,)) as cursor:
                memory = await cursor.fetchone()
            
            if memory:
                await personal_db.execute("""
                    INSERT INTO memories (query, result, embedding)
                    VALUES (?, ?, ?)
                """, memory)
                await personal_db.commit()

    async def _resolve_content_mismatch(self, discrepancy):
        # For this example, we'll always use the master version
        # In a real-world scenario, you might want a more sophisticated resolution strategy
        async with aiosqlite.connect(self.master_db_path) as master_db, \
                   aiosqlite.connect(self.personal_db_path) as personal_db:
            async with master_db.execute("SELECT result, embedding FROM memories WHERE id = ?", 
                                         (discrepancy["master_id"],)) as cursor:
                master_memory = await cursor.fetchone()
            
            if master_memory:
                await personal_db.execute("""
                    UPDATE memories
                    SET result = ?, embedding = ?
                    WHERE id = ?
                """, (*master_memory, discrepancy["personal_id"]))
                await personal_db.commit()

    async def run_consistency_check_and_fix(self):
        discrepancies = await self.perform_consistency_check()
        if discrepancies:
            logger.info(f"Found {len(discrepancies)} discrepancies. Fixing...")
            await self.fix_discrepancies(discrepancies)
            logger.info("Discrepancies fixed.")
        else:
            logger.info("No discrepancies found.")