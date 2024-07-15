from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from app.config import config
import numpy as np
from together import Together

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
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.together_client = Together(api_key=config.TOGETHER_API_KEY)

    async def add_memory(self, content: str):
        embedding = self._get_embedding(content)
        session = self.Session()
        new_memory = Memory(content=content, embedding=','.join(map(str, embedding)))
        session.add(new_memory)
        session.commit()
        await self._update_links(new_memory)
        session.close()

    async def get_relevant_memories(self, query: str, top_k: int = 5):
        query_embedding = self._get_embedding(query)
        session = self.Session()
        memories = session.query(Memory).all()
        similarities = [self._cosine_similarity(query_embedding, np.fromstring(m.embedding, sep=',')) for m in memories]
        sorted_memories = sorted(zip(memories, similarities), key=lambda x: x[1], reverse=True)[:top_k]
        session.close()
        return [(m.content, sim) for m, sim in sorted_memories]

    async def _update_links(self, new_memory: Memory):
        session = self.Session()
        all_memories = session.query(Memory).all()
        for memory in all_memories:
            if memory.id != new_memory.id:
                similarity = self._cosine_similarity(
                    np.fromstring(new_memory.embedding, sep=','),
                    np.fromstring(memory.embedding, sep=',')
                )
                if similarity > 0.8:  # Threshold for linking
                    new_memory.links.append(memory)
        session.commit()
        session.close()

    def _get_embedding(self, text: str) -> list:
        response = self.together_client.embeddings.create(
            input=[text],
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
