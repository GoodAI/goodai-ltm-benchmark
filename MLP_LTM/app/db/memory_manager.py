from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Table, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from app.config import config
import numpy as np
from openai import AsyncOpenAI
from app.utils.logging import get_logger
from contextlib import contextmanager
import os
from datetime import datetime

logger = get_logger(__name__)

Base = declarative_base()

memory_links = Table('memory_links', Base.metadata,
    Column('id', Integer, primary_key=True),
    Column('source_id', Integer, ForeignKey('memories.id')),
    Column('target_id', Integer, ForeignKey('memories.id'))
)

class Memory(Base):
    __tablename__ = 'memories'

    id = Column(Integer, primary_key=True)
    query = Column(String)
    response = Column(String)
    query_embedding = Column(String)  # Store as comma-separated string
    response_embedding = Column(String)  # Store as comma-separated string
    timestamp = Column(String)  # Store as string in format '%Y-%m-%d %H:%M:%S'

    links = relationship('Memory', secondary=memory_links,
                         primaryjoin=id==memory_links.c.source_id,
                         secondaryjoin=id==memory_links.c.target_id)

class MemoryManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        logger.info("MemoryManager initialized with OpenAI embeddings")
        
    def initialize(self):
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database initialized")

    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def create_memory_with_query(self, query: str):
        logger.debug(f"Creating new memory with query: {query[:50]}...")
        query_embedding = await self._get_embedding(query)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self.get_db() as db:
            new_memory = Memory(
                query=query,
                query_embedding=','.join(map(str, query_embedding)),
                timestamp=current_time
            )
            db.add(new_memory)
            db.commit()
            db.refresh(new_memory)
            memory_id = new_memory.id
        logger.info(f"Memory created with query: ID {memory_id}")
        return memory_id
    
    async def update_memory_with_response(self, memory_id: int, response: str):
        logger.debug(f"Updating memory ID {memory_id} with response: {response[:50]}...")
        response_embedding = await self._get_embedding(response)
        with self.get_db() as db:
            memory = db.query(Memory).filter(Memory.id == memory_id).first()
            if memory is None:
                raise ValueError(f"No memory found with ID {memory_id}")
            memory.response = response
            memory.response_embedding = ','.join(map(str, response_embedding))
            if config.MEMORY_LINKING['enabled'] and not config.MEMORY_LINKING['query_only_linking']:
                self._update_links(db, memory)
            db.commit()
        logger.info(f"Memory ID {memory_id} updated with response")

    async def get_relevant_memories(self, query: str, top_k: int = None, include_linked: bool = True):
        logger.debug(f"Retrieving relevant memories for query: {query[:50]}...")
        query_embedding = await self._get_embedding(query)
        with self.get_db() as db:
            memories = db.query(Memory).all()
            # Calculate similarities using both query and response embeddings
            similarities = [
                max(
                    self._cosine_similarity(query_embedding, np.fromstring(m.query_embedding, sep=',')),
                    self._cosine_similarity(query_embedding, np.fromstring(m.response_embedding, sep=',')) if m.response_embedding else 0
                )
                for m in memories
            ]
            sorted_memories = sorted(zip(memories, similarities), key=lambda x: x[1], reverse=True)
            
            result = [(m, sim) for m, sim in sorted_memories if sim >= config.RETRIEVAL['min_similarity']]
            
            if include_linked:
                linked_memories = set()
                for memory, _ in result:
                    for linked_memory in memory.links:
                        if linked_memory not in [m for m, _ in result]:
                            linked_sim = max(
                                self._cosine_similarity(query_embedding, np.fromstring(linked_memory.query_embedding, sep=',')),
                                self._cosine_similarity(query_embedding, np.fromstring(linked_memory.response_embedding, sep=',')) if linked_memory.response_embedding else 0
                            )
                            if linked_sim >= config.RETRIEVAL['min_similarity']:
                                linked_memories.add((linked_memory, linked_sim))
                
                result.extend(linked_memories)
            
            # Remove duplicates and sort by timestamp (most recent first)
            unique_memories = {}
            for memory, sim in result:
                if memory.id not in unique_memories:
                    unique_memories[memory.id] = memory
            
            sorted_unique_memories = sorted(unique_memories.values(), key=lambda m: m.timestamp, reverse=True)
            
            # Format memories
            formatted_memories = []
            for memory in sorted_unique_memories:
                try:
                    # Assuming memory.timestamp is stored as a string in the format '%Y-%m-%d %H:%M:%S'
                    timestamp = datetime.strptime(memory.timestamp, '%Y-%m-%d %H:%M:%S')
                    formatted_timestamp = timestamp.strftime(config.MEMORY_FORMATTING['timestamp_format'])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid timestamp format for memory ID {memory.id}: {memory.timestamp}")
                    formatted_timestamp = "Unknown Time"

                formatted_memory = f"{formatted_timestamp} {memory.query}:{memory.response}" if memory.response else f"{formatted_timestamp} {memory.query}:"
                formatted_memories.append(formatted_memory)
            
            total_matching_memories = len(formatted_memories)
            
            # Correct handling of top_k
            if top_k is None:
                top_k = config.RETRIEVAL['top_k']
            
            if top_k is not None:
                formatted_memories = formatted_memories[:top_k]
            
            logger.info(f"Retrieved {len(formatted_memories)} out of {total_matching_memories} relevant unique memories")
            
            if config.MEMORY_LINKING['enabled'] and config.MEMORY_LINKING['query_only_linking']:
                self._update_links_for_query(db, query_embedding, result)
            
            return formatted_memories

    def _update_links(self, db: Session, memory: Memory):
        logger.debug(f"Updating links for memory: ID {memory.id}")
        all_memories = db.query(Memory).filter(Memory.id != memory.id).all()
        
        links_count = 0
        for other_memory in all_memories:
            if self._should_link(memory, other_memory):
                memory.links.append(other_memory)
                links_count += 1
                logger.debug(f"Linked memory ID {other_memory.id} to memory ID {memory.id}")
            
            if config.MEMORY_LINKING['max_links_per_memory'] is not None and links_count >= config.MEMORY_LINKING['max_links_per_memory']:
                break
        
        db.commit()

    def _update_links_for_query(self, db: Session, query_embedding: np.ndarray, relevant_memories: list):
        logger.debug("Updating links for query-based retrieval")
        for memory, similarity in relevant_memories:
            if similarity > config.MEMORY_LINKING['similarity_threshold']:
                # Here you would implement the logic to create links based on the query
                # This might involve creating a temporary memory object for the query
                # or updating the links of the retrieved memories
                pass

    def _should_link(self, memory1: Memory, memory2: Memory) -> bool:
        # Calculate similarities using both query and response embeddings
        query_similarity = self._cosine_similarity(
            np.fromstring(memory1.query_embedding, sep=','),
            np.fromstring(memory2.query_embedding, sep=',')
        )
        response_similarity = 0
        if memory1.response_embedding and memory2.response_embedding:
            response_similarity = self._cosine_similarity(
                np.fromstring(memory1.response_embedding, sep=','),
                np.fromstring(memory2.response_embedding, sep=',')
            )
        
        max_similarity = max(query_similarity, response_similarity)
        
        if max_similarity > config.MEMORY_LINKING['similarity_threshold']:
            return True
        
        if config.MEMORY_LINKING['keyword_matching']['enabled']:
            # Implement keyword matching logic here
            # This is a placeholder and should be replaced with actual implementation
            keyword_similarity = self._keyword_similarity(memory1.query + memory1.response, memory2.query + memory2.response)
            if keyword_similarity > config.MEMORY_LINKING['keyword_matching']['threshold']:
                return True
        
        return False

    def _keyword_similarity(self, content1: str, content2: str) -> float:
        # This is a placeholder function for keyword matching
        # Implement your keyword matching algorithm here
        # Return a similarity score between 0 and 1
        return 0.0

    async def _get_embedding(self, text: str) -> np.ndarray:
        logger.debug("Generating embedding using OpenAI")
        try:
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=config.EMBEDDING['model']
            )
            embedding = response.data[0].embedding
            logger.debug("Embedding generated successfully")
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))