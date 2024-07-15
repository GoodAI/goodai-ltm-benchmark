import aiosqlite # type: ignore
from typing import List, Tuple, Union, Dict, Optional
from langchain_openai import OpenAIEmbeddings
import numpy as np
from config import Config
import os
from src.utils.enhanced_logging import log_execution_time
from src.utils.visualizer import visualize_memory_network
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi # type: ignore
from datetime import datetime
import time
from scipy.spatial.distance import cosine
from src.utils.structured_logging import get_logger
from src.utils.error_handling import log_error_with_traceback
import json

class MemoryManager:
    def __init__(self, openai_api_key: str):
        self.logger = get_logger('memory')
        self.config = Config()
        self.personal_db_path = self.get_personal_db_path()
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.container_id = self.get_container_id()
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []

    async def initialize(self):
        os.makedirs("/app/data", exist_ok=True)
        await self.create_tables(self.personal_db_path)
        await self._load_corpus()
        await self._update_indexing()

    def get_container_id(self):
        return os.environ.get('HOSTNAME', 'local')

    def get_personal_db_path(self):
        container_id = self.get_container_id()
        return f"/app/data/{container_id}.db"

    async def create_tables(self, db_path):
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_shape TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create memory_links table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    id INTEGER PRIMARY KEY,
                    source_memory_id INTEGER,
                    target_memory_id INTEGER,
                    link_type TEXT,
                    FOREIGN KEY (source_memory_id) REFERENCES memories (id),
                    FOREIGN KEY (target_memory_id) REFERENCES memories (id)
                )
            """)
            await db.commit()
        self.logger.debug(f"Ensured memories and memory_links tables exist in {db_path}")

    @log_execution_time
    async def create_memory_link(self, source_id: int, target_id: int, link_type: str):
        async with aiosqlite.connect(self.personal_db_path) as db:
            await db.execute("""
                INSERT INTO memory_links (source_memory_id, target_memory_id, link_type)
                VALUES (?, ?, ?)
            """, (source_id, target_id, link_type))
            await db.commit()

    @log_execution_time
    async def get_linked_memories(self, memory_id: int) -> List[Tuple[int, str]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("""
                SELECT target_memory_id, link_type FROM memory_links
                WHERE source_memory_id = ?
            """, (memory_id,)) as cursor:
                return await cursor.fetchall()


    @log_execution_time
    async def save_memory(self, query: str, result: str):
        try:
            embedding = await self.embeddings.aembed_query(query)
            
            # Convert the list to a numpy array
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_shape = embedding_array.shape
            embedding_bytes = embedding_array.tobytes()
            
            async with aiosqlite.connect(self.personal_db_path) as db:
                cursor = await db.execute("""
                    INSERT INTO memories (query, result, embedding, embedding_shape) 
                    VALUES (?, ?, ?, ?)
                """, (query, result, embedding_bytes, str(embedding_shape)))
                memory_id = cursor.lastrowid
                await db.commit()
            
            # Remove this line: self.db_logger.log_access(memory_id)
            
            self.logger.debug(f"Saved memory for query: {query} with result: {result}")
            
            # Update corpus and indexing
            await self._load_corpus()
            await self._update_indexing()

            # Call auto_link_memories with the new memory_id
            await self.auto_link_memories(memory_id)
            
            self.logger.info("Memory saved", extra={"query": query, "memory_id": memory_id})
        except Exception as e:
            log_error_with_traceback(self.logger, "Error retrieving relevant memories", e)
            raise

    async def _load_corpus(self):
        start_time = time.time()
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT query, result FROM memories") as cursor:
                memories = await cursor.fetchall()
        self.corpus = [f"{query} {result}" for query, result in memories]
        self.logger.info(f"Loaded corpus with {len(self.corpus)} entries in {time.time() - start_time:.2f} seconds")
    
    @log_execution_time
    async def _update_indexing(self):
        if self.corpus:
            start_time = time.time()
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(self.corpus)
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.logger.info(f"Updated indexing with {len(self.corpus)} documents in {time.time() - start_time:.2f} seconds")
        else:
            self.logger.warning("Corpus is empty. Skipping indexing update.")


    @log_execution_time
    async def retrieve_relevant_memories(self, query: str, threshold: float = 0.75) -> str:
        query_embedding = await self.embeddings.aembed_query(query)
        
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT id, query, result, embedding, timestamp FROM memories") as cursor:
                all_memories = await cursor.fetchall()
        
        relevant_memories = self._calculate_cosine_similarity(all_memories, query_embedding, threshold)
        
        structured_memories = {}
        for memory in relevant_memories:
            memory_id, query, result, similarity, timestamp = memory
            linked_memories = await self.get_linked_memories(int(memory_id))
            structured_memories[str(memory_id)] = {
                "Time": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                "Query": query,
                "Response": result,
                "Similarity": similarity,
                "linked_UIDs": [str(linked_id) for linked_id, _ in linked_memories]
            }
        
        # Fetch details of linked memories
        for memory_id, memory_data in structured_memories.items():
            for linked_id in memory_data["linked_UIDs"]:
                if linked_id not in structured_memories:
                    linked_memory = await self.get_memory(int(linked_id))
                    structured_memories[linked_id] = {
                        "Time": linked_memory['timestamp'].isoformat() if isinstance(linked_memory['timestamp'], datetime) else linked_memory['timestamp'],
                        "Query": linked_memory['query'],
                        "Response": linked_memory['result'],
                        "Similarity": 1.0,  # Linked memories are considered fully relevant
                        "linked_UIDs": []  # We don't fetch nested links to avoid potential infinite recursion
                    }
        
        self.logger.info("Retrieved relevant memories", 
                         extra={
                             "query": query, 
                             "total_memories": len(structured_memories),
                             "linked_memories": sum(len(m["linked_UIDs"]) for m in structured_memories.values())
                         })
        
        return json.dumps({"Relevant Injected Content": structured_memories}, indent=2)


    def _calculate_l2_norm(self, memories: List[Tuple], query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        return [
            (str(memory[0]), memory[1], memory[2], np.linalg.norm(query_embedding - np.frombuffer(memory[3])), memory[4])
            for memory in memories
            if np.linalg.norm(query_embedding - np.frombuffer(memory[3])) <= threshold
        ]

    def _calculate_cosine_similarity(self, memories: List[Tuple], query_embedding: Union[List[float], np.ndarray], threshold: float) -> List[Tuple[int, str, str, float, datetime]]:
        try:
            results = []
        
            # Ensure query_embedding is a numpy array
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            for memory in memories:
                memory_id, query, result, embedding_bytes, timestamp = memory
                memory_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Reshape memory_embedding if necessary
                if memory_embedding.shape != query_embedding.shape:
                    self.logger.warning(f"Reshaping embedding for memory {memory_id}: {memory_embedding.shape} to {query_embedding.shape}")
                    try:
                        memory_embedding = memory_embedding.reshape(query_embedding.shape)
                    except ValueError as v:
                        log_error_with_traceback(self.logger, "Cannot reshape embedding for memory {memory_id}. Skipping.", v)
                        continue
                
                # Compute cosine similarity
                similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
                
                if similarity >= threshold:
                    results.append((memory_id, query, result, float(similarity), timestamp))
 
            return results
        except Exception as e:
            log_error_with_traceback(self.logger, "Error calculating cosine similarity", e)
            raise

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
        self.logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    async def get_all_memories(self) -> List[Dict]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT id, query, result, embedding FROM memories") as cursor:
                rows = await cursor.fetchall()
                return [{
                    'id': row[0],
                    'query': row[1],
                    'result': row[2],
                    'embedding': np.frombuffer(row[3], dtype=np.float32)
                } for row in rows]

    async def _add_to_personal_db(self, memory: Tuple):
            query, result, embedding = memory
            async with aiosqlite.connect(self.personal_db_path) as db:
                await db.execute("""
                    INSERT INTO memories (query, result, embedding) VALUES (?, ?, ?)
                """, (query, result, embedding))
                await db.commit()
            self.logger.info(f"Added missing memory to personal DB: {query}")

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
            log_error_with_traceback(self.logger, "Error getting memory stats: {str(e)}", e)
            raise
        
    @log_execution_time
    async def auto_link_memories(self, new_memory_id: int, threshold: float = 0.8):
        new_memory = await self.get_memory(new_memory_id)
        all_memories = await self.get_all_memories()
        
        for memory in all_memories:
            if memory['id'] != new_memory_id:
                embedding_similarity = self._calculate_embedding_similarity(new_memory['embedding'], memory['embedding'])
                keyword_similarity = self._calculate_keyword_similarity(new_memory['query'], memory['query'])
                
                combined_similarity = (embedding_similarity + keyword_similarity) / 2
                
                if combined_similarity >= threshold:
                    await self.create_memory_link(new_memory_id, memory['id'], "auto_similar")
                    await self.create_memory_link(memory['id'], new_memory_id, "auto_similar")

    async def get_memory(self, memory_id: int) -> Dict:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT id, query, result, embedding, timestamp FROM memories WHERE id = ?", (memory_id,)) as cursor:
                row = await cursor.fetchone()
                return {
                    'id': row[0],
                    'query': row[1],
                    'result': row[2],
                    'embedding': np.frombuffer(row[3], dtype=np.float32),
                    'timestamp': row[4]
                }

    async def get_all_memories(self) -> List[Dict]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT id, query, result, embedding FROM memories") as cursor:
                rows = await cursor.fetchall()
                return [{
                    'id': row[0],
                    'query': row[1],
                    'result': row[2],
                    'embedding': np.frombuffer(row[3], dtype=np.float32)
                } for row in rows]

    def _calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return 1 - cosine(emb1, emb2)

    def _calculate_keyword_similarity(self, query1: str, query2: str) -> float:
        tokens1 = set(query1.lower().split())
        tokens2 = set(query2.lower().split())
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)
    
    async def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM memories") as cursor:
                total_memories = (await cursor.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM memory_links") as cursor:
                total_links = (await cursor.fetchone())[0]
            async with db.execute("SELECT AVG(link_count) FROM (SELECT COUNT(*) as link_count FROM memory_links GROUP BY source_memory_id)") as cursor:
                avg_links_per_memory = (await cursor.fetchone())[0]
        
        return {
            "total_memories": total_memories,
            "total_links": total_links,
            "avg_links_per_memory": avg_links_per_memory
        }
    
    async def visualize_network(self):
        memories = await self.get_all_memories()
        links = await self.get_all_links()
        
        memory_tuples = [(m['id'], m['query'], m['result']) for m in memories]
        visualize_memory_network(memory_tuples, links)

    async def get_all_links(self) -> List[Tuple[int, int, str]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT source_memory_id, target_memory_id, link_type FROM memory_links") as cursor:
                return await cursor.fetchall()