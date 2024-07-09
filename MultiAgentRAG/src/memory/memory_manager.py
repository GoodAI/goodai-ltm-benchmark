import logging
import aiosqlite # type: ignore
from typing import List, Tuple, Union, Dict, Optional
from langchain_openai import OpenAIEmbeddings
import numpy as np
from config import Config
import os
from src.utils.enhanced_logging import log_execution_time, DatabaseLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi # type: ignore
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import json

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, api_key: str):
        self.config = Config()
        self.master_db_path = "/app/data/master.db"
        self.personal_db_path = self.get_personal_db_path()
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.container_id = self.get_container_id()
        self.db_logger = DatabaseLogger(logging.getLogger('database'))
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []

    def get_container_id(self):
        return os.environ.get('HOSTNAME', 'local')

    def get_personal_db_path(self):
        container_id = self.get_container_id()
        return f"/app/data/{container_id}.db"

    async def initialize(self):
        await self.initialize_databases()
        await self._load_corpus()
        self._update_indexing()

    async def initialize_databases(self):
        os.makedirs("/app/data", exist_ok=True)
        await self.create_tables(self.master_db_path)
        await self.create_tables(self.personal_db_path)
        await self.create_changelog_table(self.master_db_path)
        await self.create_changelog_table(self.personal_db_path)

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

    @log_execution_time(logger)
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
            
            # Update corpus and indexing
            await self._load_corpus()
            self._update_indexing()
        except Exception as e:
            logger.error(f"Error saving memory for query '{query}': {str(e)}", exc_info=True)
            raise

    async def _load_corpus(self):
        start_time = time.time()
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT query, result FROM memories") as cursor:
                memories = await cursor.fetchall()
        self.corpus = [f"{query} {result}" for query, result in memories]
        logger.info(f"Loaded corpus with {len(self.corpus)} entries in {time.time() - start_time:.2f} seconds")

    def _update_indexing(self):
        if self.corpus:
            start_time = time.time()
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(self.corpus)
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Updated indexing with {len(self.corpus)} documents in {time.time() - start_time:.2f} seconds")
        else:
            logger.warning("Corpus is empty. Skipping indexing update.")

    @log_execution_time(logger)
    async def retrieve_relevant_memories(self, query: str, threshold: float = 0.75, return_metadata: bool = False) -> Union[str, Dict]:
        if not self.corpus:
            logger.warning("Corpus is empty. No memories to retrieve.")
            return ""

        start_time = time.time()
        query_embedding = np.array(await self.embeddings.aembed_query(query))
        
        try:
            async with aiosqlite.connect(self.personal_db_path) as db:
                async with db.execute("SELECT rowid, query, result, embedding, timestamp FROM memories") as cursor:
                    all_memories = await cursor.fetchall()
        except aiosqlite.Error as e:
            logger.error(f"Database error: {e}")
            return ""

        logger.debug(f"Retrieved {len(all_memories)} memories from database")

        with ThreadPoolExecutor(max_workers=4) as executor:
            l2_future = executor.submit(self._calculate_l2_norm, all_memories, query_embedding, threshold)
            cosine_future = executor.submit(self._calculate_cosine_similarity, all_memories, query_embedding, threshold)
            bm25_future = executor.submit(self._calculate_bm25, all_memories, query, threshold)
            jaccard_future = executor.submit(self._calculate_jaccard_similarity, all_memories, query, threshold)

            l2_memories = l2_future.result()
            cosine_memories = cosine_future.result()
            bm25_memories = bm25_future.result()
            jaccard_memories = jaccard_future.result()

        logger.debug(f"L2 memories: {len(l2_memories)}, Cosine memories: {len(cosine_memories)}, BM25 memories: {len(bm25_memories)}, Jaccard memories: {len(jaccard_memories)}")

        all_memories = {
            'L2 norm': l2_memories,
            'Cosine Similarity': cosine_memories,
            'BM25': bm25_memories,
            'Jaccard Similarity': jaccard_memories
        }

        formatted_output = self._format_memories(all_memories)

        logger.info(f"Retrieved and processed memories in {time.time() - start_time:.2f} seconds")

        if return_metadata:
            metadata = {
                "l2_count": len(l2_memories),
                "cosine_count": len(cosine_memories),
                "bm25_count": len(bm25_memories),
                "jaccard_count": len(jaccard_memories),
                "total_memories": len(all_memories),
                "processing_time": time.time() - start_time
            }
            return formatted_output, metadata
        else:
            return formatted_output

    def _calculate_l2_norm(self, memories: List[Tuple], query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        return [
            (str(memory[0]), memory[1], memory[2], np.linalg.norm(query_embedding - np.frombuffer(memory[3])), memory[4])
            for memory in memories
            if np.linalg.norm(query_embedding - np.frombuffer(memory[3])) <= threshold
        ]

    def _calculate_cosine_similarity(self, memories: List[Tuple], query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        return [
            (str(memory[0]), memory[1], memory[2], 
             np.dot(query_embedding, np.frombuffer(memory[3])) / (np.linalg.norm(query_embedding) * np.linalg.norm(np.frombuffer(memory[3]))),
             memory[4])
            for memory in memories
            if np.dot(query_embedding, np.frombuffer(memory[3])) / (np.linalg.norm(query_embedding) * np.linalg.norm(np.frombuffer(memory[3]))) >= threshold
        ]

    def _calculate_bm25(self, memories: List[Tuple], query: str, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        max_score = np.max(scores) if scores.size > 0 else 1
        normalized_scores = scores / max_score

        return [
            (str(memory[0]), memory[1], memory[2], score, memory[4])
            for memory, score in zip(memories, normalized_scores)
            if score >= threshold
        ]
    
    def _calculate_jaccard_similarity(self, memories: List[Tuple], query: str, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        query_set = set(query.lower().split())
        return [
            (str(memory[0]), memory[1], memory[2],
             len(set(memory[1].lower().split()) & query_set) / len(set(memory[1].lower().split()) | query_set),
             memory[4])
            for memory in memories
            if len(set(memory[1].lower().split()) & query_set) / len(set(memory[1].lower().split()) | query_set) >= threshold
        ]

    def _format_memories(self, all_memories: Dict[str, List[Tuple[str, str, str, float, str]]]) -> str:
        formatted_output = []
        seen_memories = set()

        for metric, memories in all_memories.items():
            if not memories:
                continue

            formatted_output.append(f"Similar by {metric} (ordered by timestamp - ascending):")
            sorted_memories = sorted(memories, key=lambda x: datetime.fromisoformat(x[4]))

            for memory in sorted_memories:
                memory_id, query, result, score, timestamp = memory
                if memory_id not in seen_memories:
                    formatted_output.append(f"<{memory_id}>, <{query}>, <{result}>, <{timestamp}>, <{score:.2f}>")
                    seen_memories.add(memory_id)
                else:
                    formatted_output.append(f"<{memory_id}>, <{score:.2f}>")

        return " ".join(formatted_output)

    async def get_memories(self, limit: int = 10) -> List[Tuple[str, str]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("""
                SELECT query, result FROM memories ORDER BY timestamp DESC LIMIT ?
            """, (limit,)) as cursor:
                memories = await cursor.fetchall()
        logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    async def run_consistency_check_and_fix(self):
        logger.info("Starting consistency check and fix")
        try:
            personal_memories = await self._get_all_memories(self.personal_db_path)
            master_memories = await self._get_all_memories(self.master_db_path, self.container_id)

            # Check for memories in personal DB that are not in master DB
            for memory in personal_memories:
                if memory not in master_memories:
                    await self._add_to_master_db(memory)

            # Check for memories in master DB that are not in personal DB
            for memory in master_memories:
                if memory not in personal_memories:
                    await self._add_to_personal_db(memory)

            logger.info("Consistency check and fix completed successfully")
        except Exception as e:
            logger.error(f"Error during consistency check and fix: {str(e)}", exc_info=True)
            raise

    async def _get_all_memories(self, db_path: str, author: str = None) -> List[Tuple]:
        async with aiosqlite.connect(db_path) as db:
            if author:
                async with db.execute("SELECT query, result, embedding FROM memories WHERE author = ?", (author,)) as cursor:
                    return await cursor.fetchall()
            else:
                async with db.execute("SELECT query, result, embedding FROM memories") as cursor:
                    return await cursor.fetchall()

    async def _add_to_master_db(self, memory: Tuple):
        query, result, embedding = memory
        async with aiosqlite.connect(self.master_db_path) as db:
            await db.execute("""
                INSERT INTO memories (query, result, embedding, author) VALUES (?, ?, ?, ?)
            """, (query, result, embedding, self.container_id))
            await db.commit()
        logger.info(f"Added missing memory to master DB: {query}")

    async def _add_to_personal_db(self, memory: Tuple):
            query, result, embedding = memory
            async with aiosqlite.connect(self.personal_db_path) as db:
                await db.execute("""
                    INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)
                """, (query, result, embedding))
                await db.commit()
            logger.info(f"Added missing memory to personal DB: {query}")

    async def analyze_memory_distribution(self) -> Dict[str, int]:
        """
        Analyze the distribution of memories across different similarity metrics.
        """
        distribution = {
            "L2 norm": 0,
            "Cosine Similarity": 0,
            "BM25": 0,
            "Jaccard Similarity": 0
        }

        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT query, result FROM memories") as cursor:
                memories = await cursor.fetchall()

        for memory in memories:
            query, result = memory
            _, metadata = await self.retrieve_relevant_memories(query, return_metadata=True)
            distribution["L2 norm"] += metadata["l2_count"]
            distribution["Cosine Similarity"] += metadata["cosine_count"]
            distribution["BM25"] += metadata["bm25_count"]
            distribution["Jaccard Similarity"] += metadata["jaccard_count"]

        total = sum(distribution.values())
        for key in distribution:
            distribution[key] = round((distribution[key] / total) * 100, 2) if total > 0 else 0

        return distribution

    async def prune_memories(self, threshold: float = 0.5, max_memories: int = 1000):
        """
        Prune less relevant memories to maintain system performance.
        """
        logger.info(f"Starting memory pruning process. Threshold: {threshold}, Max memories: {max_memories}")
        
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM memories") as cursor:
                (total_memories,) = await cursor.fetchone()

            if total_memories <= max_memories:
                logger.info(f"Total memories ({total_memories}) do not exceed the maximum limit. No pruning needed.")
                return

            # Retrieve all memories
            async with db.execute("SELECT id, query, result, embedding FROM memories") as cursor:
                all_memories = await cursor.fetchall()

            # Calculate relevance scores
            relevance_scores = []
            for memory in all_memories:
                memory_id, query, result, embedding = memory
                embedding_array = np.frombuffer(embedding)
                
                # Use a combination of metrics to determine relevance
                l2_norm = np.linalg.norm(embedding_array)
                cosine_sim = np.dot(embedding_array, embedding_array) / (np.linalg.norm(embedding_array) ** 2)
                
                # You might want to adjust these weights based on your specific use case
                relevance_score = 0.5 * (1 / (1 + l2_norm)) + 0.5 * cosine_sim
                
                relevance_scores.append((memory_id, relevance_score))

            # Sort memories by relevance score
            relevance_scores.sort(key=lambda x: x[1], reverse=True)

            # Determine which memories to keep
            memories_to_keep = set(score[0] for score in relevance_scores[:max_memories])

            # Delete memories that don't meet the threshold or exceed the maximum limit
            deleted_count = 0
            async with db.cursor() as cursor:
                for memory_id, relevance_score in relevance_scores[max_memories:]:
                    if relevance_score < threshold:
                        await cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                        deleted_count += 1

            await db.commit()

        logger.info(f"Memory pruning completed. Deleted {deleted_count} memories.")
        
        # Update corpus and indexing after pruning
        await self._load_corpus()
        self._update_indexing()

    async def export_memories(self, file_path: str):
        """
        Export all memories to a JSON file.
        """
        try:
            async with aiosqlite.connect(self.personal_db_path) as db:
                async with db.execute("SELECT query, result, timestamp FROM memories") as cursor:
                    memories = await cursor.fetchall()

            export_data = [
                {
                    "query": memory[0],
                    "result": memory[1],
                    "timestamp": memory[2]
                } for memory in memories
            ]

            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Successfully exported {len(memories)} memories to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting memories: {str(e)}", exc_info=True)
            raise

    async def import_memories(self, file_path: str):
        """
        Import memories from a JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)

            async with aiosqlite.connect(self.personal_db_path) as db:
                for memory in import_data:
                    query = memory['query']
                    result = memory['result']
                    timestamp = memory['timestamp']
                    embedding = await self.embeddings.aembed_query(query)
                    
                    await db.execute("""
                        INSERT INTO memories (query, result, embedding, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (query, result, np.array(embedding).tobytes(), timestamp))
                
                await db.commit()

            logger.info(f"Successfully imported {len(import_data)} memories from {file_path}")
            
            # Update corpus and indexing after import
            await self._load_corpus()
            self._update_indexing()
        except Exception as e:
            logger.error(f"Error importing memories: {str(e)}", exc_info=True)
            raise

    async def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the current state of memories.
        """
        try:
            async with aiosqlite.connect(self.personal_db_path) as db:
                async with db.execute("SELECT COUNT(*) FROM memories") as cursor:
                    (total_memories,) = await cursor.fetchone()
                
                async with db.execute("SELECT MIN(timestamp), MAX(timestamp) FROM memories") as cursor:
                    (oldest_memory, newest_memory) = await cursor.fetchone()

            distribution = await self.analyze_memory_distribution()

            return {
                "total_memories": total_memories,
                "oldest_memory": oldest_memory,
                "newest_memory": newest_memory,
                "distribution": distribution
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}", exc_info=True)
            raise

# Add any additional methods or error handling as needed