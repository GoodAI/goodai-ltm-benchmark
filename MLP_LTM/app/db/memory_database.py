# app/db/memory_database.py
from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from contextlib import contextmanager
from app.config import config
from app.utils.logging import get_logger
from typing import List, Optional
import numpy as np
from datetime import datetime

logger = get_logger('custom')

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

class MemoryDatabase:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def initialize(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise

    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def create_memory(self, query: str, query_embedding: np.ndarray) -> int:
        try:
            current_time = datetime.now().strftime(config.MEMORY_FORMATTING['timestamp_format'])
            with self.get_db() as db:
                new_memory = Memory(
                    query=query,
                    query_embedding=','.join(map(str, query_embedding)),
                    timestamp=current_time
                )
                db.add(new_memory)
                db.commit()
                db.refresh(new_memory)
                logger.info(f"Memory created with ID: {new_memory.id}")
                return new_memory.id
        except Exception as e:
            logger.error(f"Error creating memory: {str(e)}", exc_info=True)
            raise

    def update_memory_response(self, memory_id: int, response: str, response_embedding: np.ndarray):
        try:
            with self.get_db() as db:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory is None:
                    raise ValueError(f"No memory found with ID {memory_id}")
                memory.response = response
                memory.response_embedding = ','.join(map(str, response_embedding))
                db.commit()
                logger.info(f"Memory ID {memory_id} updated with response")
        except Exception as e:
            logger.error(f"Error updating memory response: {str(e)}", exc_info=True)
            raise

    def get_all_memories(self) -> List[Memory]:
        try:
            with self.get_db() as db:
                memories = db.query(Memory).all()
                logger.info(f"Retrieved {len(memories)} memories")
                return memories
        except Exception as e:
            logger.error(f"Error retrieving all memories: {str(e)}", exc_info=True)
            raise

    def add_link(self, source_id: int, target_id: int):
        try:
            with self.get_db() as db:
                source_memory = db.query(Memory).filter(Memory.id == source_id).first()
                target_memory = db.query(Memory).filter(Memory.id == target_id).first()
                if source_memory is None or target_memory is None:
                    raise ValueError(f"Invalid memory IDs: {source_id}, {target_id}")
                source_memory.links.append(target_memory)
                db.commit()
                logger.info(f"Link added between memories {source_id} and {target_id}")
        except Exception as e:
            logger.error(f"Error adding link between memories: {str(e)}", exc_info=True)
            raise

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        try:
            with self.get_db() as db:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory is None:
                    logger.warning(f"No memory found with ID {memory_id}")
                return memory
        except Exception as e:
            logger.error(f"Error retrieving memory with ID {memory_id}: {str(e)}", exc_info=True)
            raise