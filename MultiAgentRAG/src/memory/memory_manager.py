import logging
import aiosqlite # type: ignore
from typing import List, Tuple, Union, Dict, Optional
from langchain_openai import OpenAIEmbeddings
import numpy as np
from config import Config
import os
from src.utils.enhanced_logging import log_execution_time, DatabaseLogger
from src.utils.visualizer import visualize_memory_network
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi # type: ignore
from datetime import datetime
import time
from scipy.spatial.distance import cosine

logger = logging.getLogger('memory')

class MemoryManager:
    def __init__(self, api_key: str):
        self.config = Config()
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
        os.makedirs("/app/data", exist_ok=True)
        await self.create_tables(self.personal_db_path)
        await self._load_corpus()
        self._update_indexing()

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
        logger.debug(f"Ensured memories and memory_links tables exist in {db_path}")

    async def create_memory_link(self, source_id: int, target_id: int, link_type: str):
        async with aiosqlite.connect(self.personal_db_path) as db:
            await db.execute("""
                INSERT INTO memory_links (source_memory_id, target_memory_id, link_type)
                VALUES (?, ?, ?)
            """, (source_id, target_id, link_type))
            await db.commit()

    async def get_linked_memories(self, memory_id: int) -> List[Tuple[int, str]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("""
                SELECT target_memory_id, link_type FROM memory_links
                WHERE source_memory_id = ?
            """, (memory_id,)) as cursor:
                return await cursor.fetchall()


    @log_execution_time(logger)
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
            
            self.db_logger.log_access(memory_id)
            
            logger.debug(f"Saved memory for query: {query} with result: {result}")
            
            # Update corpus and indexing
            await self._load_corpus()
            self._update_indexing()

            # Call auto_link_memories with the new memory_id
            await self.auto_link_memories(memory_id)
            
            return memory_id
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
    async def retrieve_relevant_memories(self, query: str, threshold: float = 0.75) -> str:
        query_embedding = await self.embeddings.aembed_query(query)
        
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT id, query, result, embedding FROM memories") as cursor:
                all_memories = await cursor.fetchall()
        
        relevant_memories = self._calculate_cosine_similarity(all_memories, query_embedding, threshold)
        
        # Retrieve linked memories
        linked_memories = set()
        for memory in relevant_memories:
            memory_id = memory[0]  # Assuming the first element is the memory_id
            linked = await self.get_linked_memories(int(memory_id))
            linked_memories.update(linked)
        
        # Fetch details of linked memories
        for linked_id, link_type in linked_memories:
            memory = await self.get_memory(linked_id)
            relevant_memories.append((str(memory['id']), memory['query'], memory['result'], 1.0, f"Linked ({link_type})"))
        
        # Sort memories by similarity
        relevant_memories.sort(key=lambda x: x[3], reverse=True)
        
        # Format the output
        formatted_output = []
        for memory in relevant_memories:
            if len(memory) == 5:
                memory_id, query, result, similarity, link_info = memory
                formatted_output.append(f"<{memory_id}>, <{query}>, <{result}>, <{similarity:.2f}>, <{link_info}>")
            else:
                memory_id, query, result, similarity = memory
                formatted_output.append(f"<{memory_id}>, <{query}>, <{result}>, <{similarity:.2f}>")
        
        return " ".join(formatted_output)

    def _calculate_l2_norm(self, memories: List[Tuple], query_embedding: np.ndarray, threshold: float) -> List[Tuple[str, str, str, float, str]]:
        return [
            (str(memory[0]), memory[1], memory[2], np.linalg.norm(query_embedding - np.frombuffer(memory[3])), memory[4])
            for memory in memories
            if np.linalg.norm(query_embedding - np.frombuffer(memory[3])) <= threshold
        ]

    def _calculate_cosine_similarity(self, memories: List[Tuple], query_embedding: Union[List[float], np.ndarray], threshold: float) -> List[Tuple[str, str, str, float, str]]:
        results = []
        
        # Ensure query_embedding is a numpy array
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        
        for memory in memories:
            memory_id, query, result, embedding_bytes = memory
            memory_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Reshape memory_embedding if necessary
            if memory_embedding.shape != query_embedding.shape:
                logger.warning(f"Reshaping embedding for memory {memory_id}: {memory_embedding.shape} to {query_embedding.shape}")
                try:
                    memory_embedding = memory_embedding.reshape(query_embedding.shape)
                except ValueError:
                    logger.error(f"Cannot reshape embedding for memory {memory_id}. Skipping.")
                    continue
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, memory_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
            
            if similarity >= threshold:
                results.append((str(memory_id), query, result, float(similarity), ""))
        
        return results

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
            async with db.execute("SELECT id, query, result, embedding FROM memories WHERE id = ?", (memory_id,)) as cursor:
                row = await cursor.fetchone()
                return {
                    'id': row[0],
                    'query': row[1],
                    'result': row[2],
                    'embedding': np.frombuffer(row[3], dtype=np.float32)
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

    async def analyze_link_distribution(self) -> Dict[str, int]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT link_type, COUNT(*) FROM memory_links GROUP BY link_type") as cursor:
                distribution = dict(await cursor.fetchall())
        return distribution
    
    async def visualize_network(self):
        memories = await self.get_all_memories()
        links = await self.get_all_links()
        
        memory_tuples = [(m['id'], m['query'], m['result']) for m in memories]
        visualize_memory_network(memory_tuples, links)

    async def get_all_links(self) -> List[Tuple[int, int, str]]:
        async with aiosqlite.connect(self.personal_db_path) as db:
            async with db.execute("SELECT source_memory_id, target_memory_id, link_type FROM memory_links") as cursor:
                return await cursor.fetchall()