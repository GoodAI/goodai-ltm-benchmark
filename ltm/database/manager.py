from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.inspection import inspect
from typing import List
import json

from ltm.database.models import Base, Session, Message


class DatabaseManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionMaker = sessionmaker(bind=self.engine)

    def create_session(self, session_id, created_at):
        with self.SessionMaker() as db_session:
            new_session = Session(session_id=session_id, created_at=created_at)
            db_session.add(new_session)
            db_session.commit()
            return new_session

    def add_message(self, session_id, role, content, timestamp, semantic_key):
        with self.SessionMaker() as db_session:
            new_message = Message(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=timestamp,
                semantic_key=semantic_key
            )
            db_session.add(new_message)
            db_session.commit()
            return new_message

    def get_recent_messages(self, session_id, limit=10) -> List[Message]:
        with self.SessionMaker() as db_session:
            return db_session.query(Message).filter(Message.session_id == session_id) \
                .order_by(desc(Message.timestamp)).limit(limit).all()

    def get_messages_by_timestamp(self, session_id: str, timestamp: float) -> List[Message]:
        with self.SessionMaker() as db_session:
            messages = db_session.query(Message).filter(
                Message.session_id == session_id,
                Message.timestamp == timestamp
            ).order_by(Message.id).limit(2).all()
            return messages

    def clear_all(self):
        with self.SessionMaker() as db_session:
            db_session.query(Message).delete()
            db_session.query(Session).delete()
            db_session.commit()

    def export_data(self) -> str:
        with self.SessionMaker() as db_session:
            sessions = db_session.query(Session).all()
            messages = db_session.query(Message).all()
            data = {
                "sessions": [self.object_as_dict(s) for s in sessions],
                "messages": [self.object_as_dict(m) for m in messages]
            }
            return json.dumps(data)

    def import_data(self, data_str: str):
        data = json.loads(data_str)
        with self.SessionMaker() as db_session:
            for session_data in data["sessions"]:
                session = Session(**session_data)
                db_session.add(session)
            for message_data in data["messages"]:
                message = Message(**message_data)
                db_session.add(message)
            db_session.commit()

    @staticmethod
    def object_as_dict(obj):
        return {c.key: getattr(obj, c.key)
                for c in inspect(obj).mapper.column_attrs}

    # def get_session(self, session_id) -> Optional[Session]:
    #     with self.SessionMaker() as db_session:
    #         return db_session.query(Session).filter(Session.session_id == session_id).first()

    # def get_all_messages(self, session_id) -> List[Message]:
    #     with self.SessionMaker() as db_session:
    #         return db_session.query(Message).filter(Message.session_id == session_id) \
    #             .order_by(Message.timestamp).all()

    # def get_interaction_by_semantic_key(self, semantic_key) -> Optional[Tuple[Message, Message]]:
    #     with self.SessionMaker() as db_session:
    #         messages = db_session.query(Message).filter(Message.semantic_key == semantic_key).order_by(Message.timestamp).limit(2).all()
    #         if len(messages) == 2:
    #             return (messages[0], messages[1])
    #         return None

    # def get_message_count(self, session_id) -> int:
    #     with self.SessionMaker() as db_session:
    #         return db_session.query(Message).filter(Message.session_id == session_id).count()