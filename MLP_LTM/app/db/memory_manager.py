from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Table, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from app.config import config
import numpy as np
from together import Together
from app.utils.logging import get_logger
from contextlib import contextmanager

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
    content = Column(String)
    embedding = Column(String)  # Store as comma-separated string
    timestamp = Column(Float, server_default=func.now())

    links = relationship('Memory', secondary=memory_links,
                         primaryjoin=id==memory_links.c.source_id,
                         secondaryjoin=id==memory_links.c.target_id)

class MemoryManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.together_client = Together(api_key=config.TOGETHER_API_KEY)
        logger.info("MemoryManager initialized")
        
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

    async def add_memory(self, content: str):
        logger.debug(f"Adding new memory: {content[:50]}...")  # Log first 50 chars
        embedding = await self._get_embedding(content)
        with self.get_db() as db:
            new_memory = Memory(content=content, embedding=','.join(map(str, embedding)))
            db.add(new_memory)
            db.commit()
            db.refresh(new_memory)
            memory_id = new_memory.id  # Get the id within the session
            self._update_links(db, new_memory)
        logger.info(f"Memory added successfully: ID {memory_id}")
        return memory_id

    async def get_relevant_memories(self, query: str, top_k: int = 5):
        logger.debug(f"Retrieving relevant memories for query: {query[:50]}...")
        query_embedding = await self._get_embedding(query)
        with self.get_db() as db:
            memories = db.query(Memory).all()
            similarities = [self._cosine_similarity(query_embedding, np.fromstring(m.embedding, sep=',')) for m in memories]
            sorted_memories = sorted(zip(memories, similarities), key=lambda x: x[1], reverse=True)[:top_k]
            result = [(m.content, sim) for m, sim in sorted_memories]
        logger.info(f"Retrieved {len(result)} relevant memories")
        return result

    def _update_links(self, db: Session, new_memory: Memory):
        logger.debug(f"Updating links for memory: ID {new_memory.id}")
        all_memories = db.query(Memory).filter(Memory.id != new_memory.id).all()
        
        for memory in all_memories:
            similarity = self._cosine_similarity(
                np.fromstring(new_memory.embedding, sep=','),
                np.fromstring(memory.embedding, sep=',')
            )
            if similarity > 0.8:  # Threshold for linking
                new_memory.links.append(memory)
                logger.debug(f"Linked memory ID {memory.id} to new memory ID {new_memory.id}")
        db.commit()

    async def _get_embedding(self, text: str) -> list:
        logger.debug("Generating embedding")
        response = self.together_client.embeddings.create(
            input=[text],
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        )
        logger.debug("Embedding generated successfully")
        return response.data[0].embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))