from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    created_at = Column(Float, nullable=False)

    messages = relationship("Message", back_populates="session")

    def __repr__(self):
        return f"<Session(id={self.id}, session_id={self.session_id}, created_at={self.created_at})>"


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.session_id'), nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    semantic_key = Column(String, nullable=False)

    session = relationship("Session", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, session_id={self.session_id}, role={self.role}, timestamp={self.timestamp})>"