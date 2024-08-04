import json
import uuid
from dataclasses import dataclass, field, asdict
import datetime
from math import ceil
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    Text,
    Table,
    ForeignKeyConstraint,
    Index,
)
from contextlib import contextmanager

from goodai.helpers.json_helper import (
    sanitize_and_parse_json,
    SimpleJSONEncoder,
    SimpleJSONDecoder,
)
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory
from litellm import token_counter
from model_interfaces.base_ltm_agent import Message
from utils.constants import DATA_DIR

from utils.llm import make_system_message, make_user_message, ask_llm
from utils.ui import colour_print
from utils.text import td_format
import logging

logging.basicConfig(
    filename='ltm_agent.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


_debug_dir = DATA_DIR.joinpath("ltm_debug_info")

Base = declarative_base()


class DatabaseManager:
    def __init__(self, engine):
        self.engine = engine
        self.SessionMaker = sessionmaker(bind=engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self):
        session = self.SessionMaker()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def get_or_create_keyword(self, session, keyword):
        keyword_obj = session.query(Keyword).filter_by(keyword=keyword).first()
        if not keyword_obj:
            keyword_obj = Keyword(keyword=keyword)
            session.add(keyword_obj)
            session.flush()
        return keyword_obj


class DBSession(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    messages = relationship("DBMessage", back_populates="session")


class DBMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    role = Column(String)
    content = Column(Text)
    semantic_id = Column(String, index=True)  # ID in semantic database
    session = relationship("DBSession", back_populates="messages")
    keywords = relationship("Keyword",
                            secondary="message_keywords",
                            back_populates="messages")

    __table_args__ = (Index("idx_message_timestamp_session_role", "timestamp",
                            "session_id", "role"), )


class DBInteraction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    user_message_id = Column(Integer, ForeignKey("messages.id"))
    assistant_message_id = Column(Integer, ForeignKey("messages.id"))
    user_message = relationship("DBMessage", foreign_keys=[user_message_id])
    assistant_message = relationship("DBMessage",
                                     foreign_keys=[assistant_message_id])

    __table_args__ = (Index("idx_interaction_timestamp_session", "timestamp",
                            "session_id"), )


class Keyword(Base):
    __tablename__ = "keywords"
    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True, index=True)
    messages = relationship("DBMessage", secondary="message_keywords", back_populates="keywords")


message_keywords = Table(
    "message_keywords",
    Base.metadata,
    Column("message_id", Integer, ForeignKey("messages.id"), primary_key=True),
    Column("keyword_id", Integer, ForeignKey("keywords.id"), primary_key=True),
)


@dataclass
class Memory:
    passage: str
    metadata: Dict[str, Any]
    timestamp: float
    relevance: float = 1.0 #? Potential issue. 
    semantic_id: Optional[str] = None
    relational_id: Optional[int] = None


class DualDatabaseInterface:

    def __init__(self, db_manager: DatabaseManager, semantic_memory):
        self.db_manager = db_manager
        self.semantic_memory = semantic_memory

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: float,
        keywords: List[str],
    ) -> Memory:
        with self.db_manager.session_scope() as session: #TODO all these calls can be done with a decorator
            db_message = DBMessage(timestamp=timestamp,
                                   session_id=session_id,
                                   role=role,
                                   content=content)
            session.add(db_message)
            session.flush()

            for kw in keywords:
                keyword = self.db_manager.get_or_create_keyword(session, kw)
                db_message.keywords.append(keyword)

            semantic_id = self.semantic_memory.add_text(
                content,
                timestamp=timestamp,
                metadata={
                    "keywords": keywords,
                    "role": role,
                    "session_id": session_id
                },
            )
            db_message.semantic_id = semantic_id

            memory = Memory(
                passage=content,
                metadata={
                    "keywords": keywords,
                    "role": role,
                    "session_id": session_id
                },
                timestamp=timestamp,
                semantic_id=semantic_id,
                relational_id=db_message.id,
            )
            
            logger.info(f"Message added with semantic_id: {semantic_id}")
            return memory

    def get_relevant_memories(self,
                            query: str,
                            session_id: Optional[str] = None,
                            limit: int = 10) -> List[Memory]:
        logger.info(f"Starting get_relevant_memories with query: '{query}', session_id: {session_id}, limit: {limit}")
        colour_print("CYAN", f"Starting get_relevant_memories with query: '{query}', session_id: {session_id}, limit: {limit}")
        
        try:
            colour_print("YELLOW", "Retrieving semantic results")
            semantic_results = self.semantic_memory.retrieve(query, k=limit)
            colour_print("GREEN", f"Retrieved {len(semantic_results)} semantic results")
        except Exception as e:
            colour_print("RED", f"Error in semantic retrieval: {str(e)}")
            colour_print("YELLOW", "Falling back to keyword-based retrieval")
            return self.retrieve_from_keywords(query.lower().split(), session_id, limit)

        with self.db_manager.session_scope() as session:
            colour_print("YELLOW", "Entering database session scope")
            memories = []
            for idx, result in enumerate(semantic_results, 1):
                colour_print("YELLOW", f"Processing semantic result {idx}/{len(semantic_results)}")
                try:
                    # Check if 'id' attribute exists, if not, use an alternative
                    semantic_id = getattr(result, 'id', None)
                    if semantic_id is None:
                        # If 'id' doesn't exist, try to use 'passage_id' or any other unique identifier
                        semantic_id = getattr(result, 'passage_id', str(idx))
                    
                    colour_print("GREEN", f"Using semantic_id: {semantic_id}")
                    
                    db_message = (session.query(DBMessage).filter_by(
                        semantic_id=semantic_id).first())
                    
                    if db_message:
                        colour_print("GREEN", f"Found matching DB message for semantic_id: {semantic_id}")
                        if session_id is None or db_message.session_id == session_id:
                            colour_print("GREEN", "Session ID match or not required")
                            memory = self._create_memory_from_db_message(db_message, getattr(result, 'relevance', 1.0))
                            memories.append(memory)
                            colour_print("GREEN", f"Created and appended memory with relevance: {getattr(result, 'relevance', 1.0)}")
                        else:
                            colour_print("YELLOW", f"Skipped due to session_id mismatch. Required: {session_id}, Found: {db_message.session_id}")
                    else:
                        colour_print("RED", f"No matching DB message found for semantic_id: {semantic_id}")
                except Exception as e:
                    colour_print("RED", f"Error processing semantic result: {str(e)}")

            colour_print("CYAN", f"Returning {len(memories)} memories")
            return memories
        
    def add_keywords(self, keywords: List[str]):
        with self.db_manager.session_scope() as session:
            existing_keywords = set(session.query(Keyword.keyword).filter(Keyword.keyword.in_(keywords)).all())
            new_keywords = set(keywords) - existing_keywords
            for keyword in new_keywords:
                new_keyword = Keyword(keyword=keyword)
                session.add(new_keyword)

    def get_all_keywords(self) -> List[str]:
        with self.db_manager.session_scope() as session:
            keywords = session.query(Keyword.keyword).all()
            return [keyword[0] for keyword in keywords]
        
    def retrieve_from_keywords(self, keywords: List[str], session_id: Optional[str] = None, limit: int = 10) -> List[Memory]:
        colour_print("YELLOW", f"Retrieving memories from keywords: {keywords}")
        
        with self.db_manager.session_scope() as session:
            query = session.query(DBMessage).join(DBMessage.keywords)
            if session_id:
                query = query.filter(DBMessage.session_id == session_id)
            
            query = query.filter(Keyword.keyword.in_(keywords))
            db_messages = query.limit(limit).all()
            
            memories = []
            for db_message in db_messages:
                memory = self._create_memory_from_db_message(db_message)
                memories.append(memory)
        
        colour_print("GREEN", f"Retrieved {len(memories)} memories from keywords")
        return memories
    
    def get_interactions_from_memories(self, memories: Union[Memory, List[Memory]], session_id: str) -> List[Tuple[Memory, Memory]]:
        colour_print("YELLOW", f"Getting interactions from {len(memories) if isinstance(memories, list) else 1} memories")
        interactions = []
        
        if not isinstance(memories, list):
            memories = [memories]
        
        with self.db_manager.session_scope() as session:
            for memory in memories:
                interaction = (session.query(DBInteraction)
                            .filter_by(session_id=session_id)
                            .filter((DBInteraction.user_message_id == memory.relational_id) |
                                    (DBInteraction.assistant_message_id == memory.relational_id))
                            .first())
                
                if interaction:
                    user_memory = self._create_memory_from_db_message(interaction.user_message)
                    assistant_memory = self._create_memory_from_db_message(interaction.assistant_message)                    
                    interactions.append((user_memory, assistant_memory))
        
        return interactions
    
    def _create_memory_from_db_message(self, db_message: DBMessage, relevance: float = 1.0) -> Memory:
        return Memory(
            passage=db_message.content,
            metadata={
                "keywords": [kw.keyword for kw in db_message.keywords],
                "role": db_message.role,
                "session_id": db_message.session_id,
            },
            timestamp=db_message.timestamp,
            relevance=relevance,
            semantic_id=db_message.semantic_id,
            relational_id=db_message.id,
        )

    def sync_databases(self):
        with self.db_manager.session_scope() as session:
            unsynced_messages = (session.query(DBMessage).filter(
                DBMessage.semantic_id.is_(None)).all())
            for message in unsynced_messages:
                semantic_id = self.semantic_memory.add_text(
                    message.content,
                    timestamp=message.timestamp,
                    metadata={
                        "keywords": [kw.keyword for kw in message.keywords],
                        "role": message.role,
                        "session_id": message.session_id,
                    },
                )
                message.semantic_id = semantic_id


class LTMAgentSession:
    """
    An agent session, or a collection of messages, utilizing the dual database system.
    """

    def __init__(self, session_id: str,
                 dual_db_interface: DualDatabaseInterface):
        self.session_id = session_id
        self.dual_db_interface = dual_db_interface

        with self.dual_db_interface.db_manager.session_scope() as session:
            # Create a new database session entry
            db_session = DBSession(session_id=self.session_id)
            session.add(db_session)

    @property
    def message_count(self):
        with self.dual_db_interface.db_manager.session_scope() as session:
            return (session.query(DBMessage).filter_by(
                session_id=self.session_id).count())

    def state_as_text(self) -> str:
        """
        :return: A string that represents the contents of the session.
        """
        with self.dual_db_interface.db_manager.session_scope() as session:
            messages = (session.query(DBMessage).filter_by(
                session_id=self.session_id).order_by(
                    DBMessage.timestamp).all())
            interactions = (session.query(DBInteraction).filter_by(
                session_id=self.session_id).order_by(
                    DBInteraction.timestamp).all())

            state = {
                "session_id":
                self.session_id,
                "history": [
                    asdict(
                        Memory(
                            passage=m.content,
                            metadata={
                                "role": m.role,
                                "keywords": [kw.keyword for kw in m.keywords],
                            },
                            timestamp=m.timestamp,
                            semantic_id=m.semantic_id,
                            relational_id=m.id,
                        )) for m in messages
                ],
                "interactions": {
                    str(i.timestamp): (
                        asdict(
                            Memory(
                                passage=i.user_message.content,
                                metadata={
                                    "role":
                                    i.user_message.role,
                                    "keywords": [
                                        kw.keyword
                                        for kw in i.user_message.keywords
                                    ],
                                },
                                timestamp=i.user_message.timestamp,
                                semantic_id=i.user_message.semantic_id,
                                relational_id=i.user_message.id,
                            )),
                        asdict(
                            Memory(
                                passage=i.assistant_message.content,
                                metadata={
                                    "role":
                                    i.assistant_message.role,
                                    "keywords": [
                                        kw.keyword
                                        for kw in i.assistant_message.keywords
                                    ],
                                },
                                timestamp=i.assistant_message.timestamp,
                                semantic_id=i.assistant_message.semantic_id,
                                relational_id=i.assistant_message.id,
                            )),
                    )
                    for i in interactions
                },
            }

        return json.dumps(state, cls=SimpleJSONEncoder)

    def add_interaction(self, interaction: Tuple[Message, Message],
                        keywords: List[str]):
        user_message, assistant_message = interaction

        user_memory = self.dual_db_interface.add_message(
            session_id=self.session_id,
            role=user_message.role,
            content=user_message.content,
            timestamp=user_message.timestamp,
            keywords=keywords,
        )

        assistant_memory = self.dual_db_interface.add_message(
            session_id=self.session_id,
            role=assistant_message.role,
            content=assistant_message.content,
            timestamp=assistant_message.timestamp,
            keywords=keywords,
        )

        with self.dual_db_interface.db_manager.session_scope() as session:
            db_interaction = DBInteraction(
                timestamp=user_message.timestamp,
                session_id=self.session_id,
                user_message_id=user_memory.relational_id,
                assistant_message_id=assistant_memory.relational_id,
            )
            session.add(db_interaction)

    def interaction_from_timestamp(
            self, timestamp: float) -> Optional[Tuple[Memory, Memory]]:
        with self.dual_db_interface.db_manager.session_scope() as session:
            db_interaction = (session.query(DBInteraction).filter_by(
                timestamp=timestamp, session_id=self.session_id).first())
            if db_interaction:
                user_memory = Memory(
                    passage=db_interaction.user_message.content,
                    metadata={
                        "role":
                        db_interaction.user_message.role,
                        "keywords": [
                            kw.keyword
                            for kw in db_interaction.user_message.keywords
                        ],
                    },
                    timestamp=db_interaction.user_message.timestamp,
                    semantic_id=db_interaction.user_message.semantic_id,
                    relational_id=db_interaction.user_message.id,
                )
                assistant_memory = Memory(
                    passage=db_interaction.assistant_message.content,
                    metadata={
                        "role":
                        db_interaction.assistant_message.role,
                        "keywords": [
                            kw.keyword
                            for kw in db_interaction.assistant_message.keywords
                        ],
                    },
                    timestamp=db_interaction.assistant_message.timestamp,
                    semantic_id=db_interaction.assistant_message.semantic_id,
                    relational_id=db_interaction.assistant_message.id,
                )
                return (user_memory, assistant_memory)
        return None

    def by_index(self, idx) -> Optional[Memory]:
        with self.dual_db_interface.db_manager.session_scope() as session:
            message = (session.query(DBMessage).filter_by(
                session_id=self.session_id).order_by(
                    DBMessage.timestamp).offset(idx).first())
            if message:
                return Memory(
                    passage=message.content,
                    metadata={
                        "role": message.role,
                        "keywords": [kw.keyword for kw in message.keywords],
                    },
                    timestamp=message.timestamp,
                    semantic_id=message.semantic_id,
                    relational_id=message.id,
                )
        return None

    @classmethod
    def from_state_text(
            cls, state_text: str,
            dual_db_interface: DualDatabaseInterface) -> "LTMAgentSession":
        """
        Builds a session object given state text.
        :param state_text: Text previously obtained using the state_as_text() method.
        :param dual_db_interface: DualDatabaseInterface instance for database operations.
        :return: A session instance.
        """
        state: dict = json.loads(state_text, cls=SimpleJSONDecoder)
        session_id = state["session_id"]

        session = cls(session_id, dual_db_interface)

        for memory_dict in state["history"]:
            memory = Memory(**memory_dict)
            session.dual_db_interface.add_message(
                session_id=session_id,
                role=memory.metadata["role"],
                content=memory.passage,
                timestamp=memory.timestamp,
                keywords=memory.metadata["keywords"],
            )

        for timestamp, (
                user_memory_dict,
                assistant_memory_dict) in state["interactions"].items():
            user_memory = Memory(**user_memory_dict)
            assistant_memory = Memory(**assistant_memory_dict)
            session.add_interaction(
                (
                    Message(
                        role=user_memory.metadata["role"],
                        content=user_memory.passage,
                        timestamp=user_memory.timestamp,
                    ),
                    Message(
                        role=assistant_memory.metadata["role"],
                        content=assistant_memory.passage,
                        timestamp=assistant_memory.timestamp,
                    ),
                ),
                keywords=user_memory.metadata["keywords"],
            )

        return session

    def get_interaction(self,
                        timestamp: float) -> Optional[Tuple[Memory, Memory]]:
        return self.interaction_from_timestamp(timestamp)


@dataclass
class InsertedContextAgent:
    max_completion_tokens: Optional[int] = None
    max_prompt_size: int = 16384
    is_local: bool = True
    llm_call_idx: int = 0
    costs_usd: float = 0.0
    model: str = "gpt-4o-mini"
    # model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: float = 0.01
    system_message: str = "You are a helpful AI assistant."
    debug_level: int = 1
    session: Optional[LTMAgentSession] = None
    now: Optional[datetime.datetime] = None
    db_url: str = "sqlite:///ltm_sessions.db"
    dual_db_interface: Optional[DualDatabaseInterface] = None
    run_name: str = ""

    def __post_init__(self):
        self.max_message_size = 1000
        self.init_timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S")

        if self.dual_db_interface is None:
            engine = create_engine(self.db_url)
            db_manager = DatabaseManager(engine)
            db_manager.create_tables()
            semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(
                chunk_capacity=50, chunk_overlap_fraction=0.0))
            self.dual_db_interface = DualDatabaseInterface(db_manager, semantic_memory)

        self.new_session()

    @property
    def save_name(self) -> str:
        return f"{self.model}-{self.max_prompt_size}-{self.max_completion_tokens}__{self.init_timestamp}"

    def new_session(self) -> "LTMAgentSession":
        session_id = str(uuid.uuid4())
        self.session = LTMAgentSession(
            session_id=session_id, dual_db_interface=self.dual_db_interface)
        return self.session

    def reply(
        self,
        user_message: str,
        agent_response: Optional[str] = None,
        cost_callback: Callable = None,
    ) -> str:
        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")
        self.now = datetime.datetime.now()

        keywords = self.keywords_for_message(user_message,
                                             cost_cb=cost_callback)
        context = self.create_context(
            user_message,
            max_prompt_size=self.max_prompt_size,
            previous_interactions=0,
            cost_cb=cost_callback,
        )
        response_text = ask_llm(
            context,
            model=self.model,
            max_overall_tokens=self.max_prompt_size,
            cost_callback=cost_callback,
            temperature=self.temperature,
        )
        # debug_actions(
        #     context,
        #     self.temperature,
        #     response_text,
        #     self.llm_call_idx,
        #     self.debug_level,
        #     self.save_name,
        #     name_template="reply-{idx}",
        # )
        self.llm_call_idx += 1

        self.save_interaction(user_message, response_text, keywords)

        return response_text

    def keywords_for_message(self, user_message: str, cost_cb: Callable) -> List[str]:
        defined_kws = set(self.dual_db_interface.get_all_keywords())
        prompt = f'''Create two keywords to describe the topic of this message:
    "{user_message}".

    Focus on the topic and tone of the message. Produce the keywords in JSON like: `["keyword_1", "keyword_2"]`

    Choose keywords that would aid in retrieving this message from memory in the future.

    Reuse these keywords if appropriate: {", ".join(defined_kws)}'''

        context = [make_system_message(prompt)]
        while True:
            try:
                print("Keyword gen")
                response = ask_llm(
                    context,
                    model=self.model,
                    max_overall_tokens=self.max_prompt_size,
                    cost_callback=cost_cb,
                    temperature=self.temperature,
                )
                keywords = [k.lower() for k in sanitize_and_parse_json(response)]
                break
            except Exception as e:
                print(repr(e) + response)
                continue

        new_keywords = [kw for kw in keywords if kw not in defined_kws]
        if new_keywords:
            self.dual_db_interface.add_keywords(new_keywords)
        
        print(f"Interaction keywords: {keywords}")
        return keywords

        self.dual_db_interface.add_keywords(keywords)  # Add all keywords, maintaining original functionality
        print(f"Interaction keywords: {keywords}")
        return keywords

    def create_context(
        self,
        user_message: str,
        max_prompt_size: int,
        previous_interactions: int,
        cost_cb: Callable,
    ) -> List[Dict[str, str]]:
        stamped_user_message = f"{self.now}: {user_message}"
        context = [
            make_system_message(self.system_message),
            make_user_message(stamped_user_message),
        ]
        relevant_memories = self.get_relevant_memories(user_message, cost_cb)

        full_interactions = self.dual_db_interface.get_interactions_from_memories(
            relevant_memories, self.session.session_id)

        for user_memory, assistant_memory in full_interactions:
            if "trivia" in user_memory.passage:
                colour_print("YELLOW", f"<*** trivia ***>")
            else:
                colour_print("YELLOW", f"{user_memory.passage}")

        # Add the previous messages
        final_idx = self.session.message_count - 1
        while previous_interactions > 0 and final_idx > 0:
            context.insert(
                1, self.memory_to_dict(
                    self.session.by_index(final_idx)))  # Agent reply
            context.insert(1,
                           self.memory_to_dict(
                               self.session.by_index(final_idx -
                                                     1)))  # User message
            final_idx -= 2
            previous_interactions -= 1

        # Add in memories up to the max prompt size
        current_size = token_counter(self.model, messages=context)
        shown_mems = 0
        target_size = max_prompt_size - self.max_message_size

        for user_memory, assistant_memory in full_interactions[::-1]:
            future_size = token_counter(
                self.model,
                messages=context + [
                    self.memory_to_dict(user_memory),
                    self.memory_to_dict(assistant_memory),
                ],
            )

            if shown_mems >= 100 or future_size > target_size:
                break

            context.insert(1, self.memory_to_dict(assistant_memory))
            ts = datetime.datetime.fromtimestamp(user_memory.timestamp)
            et_descriptor = f"{str(ts)[:-7]} ({td_format(self.now - ts)}) "
            user_dict = self.memory_to_dict(user_memory)
            user_dict["content"] = et_descriptor + user_dict["content"]
            context.insert(1, user_dict)

            shown_mems += 1
            current_size = future_size

        print(f"current context size: {current_size}")
        return context

    def memory_to_dict(self, memory: Memory) -> Dict[str, str]:
        return {"role": memory.metadata["role"], "content": memory.passage}

    def llm_memory_filter(
        self,
        memories: List[Memory],
        queries: List[str],
        keywords: List[str],
        cost_cb: Callable,
    ) -> List[Memory]: #TODO explore threading
        situation_prompt = """You are a part of an agent. Another part of the agent is currently searching for memories using the statements below.
Based on these statements, describe what is currently happening external to the agent in general terms:
{queries}  
"""

        prompt = """Here are a number of interactions, each is given a number:
{passages}         
*********

Each of these interactions might be related to the general situation below. Your task is to judge if these interaction have any relation to the general situation.
Filter out interactions that very clearly do not have any relation. But keep in interactions that have any kind of relationship to the situation such as in: topic, characters, locations, setting, etc.

SITUATION:
{situation}

Express your answer in this JSON: 
[
    {{
        "number": int  // The number of the interaction.
        "justification": string  // Why the interaction is or is not related to the situation.
        "related": bool // Whether the interaction is related to the situation.
    }},
    ...
]
"""

        if len(memories) == 0:
            return []

        splice_length = 10

        mems_to_filter = []  # Memories without duplicates
        filtered_mems = []

        # Get the situation #! worth making this a semantic search over situation column?  
        queries_txt = "- " + "\n- ".join(queries)
        context = [
            make_user_message(situation_prompt.format(queries=queries_txt))
        ]
        situation = ask_llm(
            context,
            model=self.model,
            max_overall_tokens=self.max_prompt_size,
            cost_callback=cost_cb,
            temperature=self.temperature,
        )
        colour_print("MAGENTA", f"Filtering situation: {situation}")

        # Remove duplicate memories
        seen_timestamps = set()
        for m in memories:
            if m.timestamp not in seen_timestamps:
                mems_to_filter.append(m)
                seen_timestamps.add(m.timestamp)

        num_splices = ceil(len(mems_to_filter) / splice_length)
        # Iterate through the mems_to_filter list and create the passage
        call_count = 0
        for splice in range(num_splices):
            start_idx = splice * splice_length
            end_idx = (splice + 1) * splice_length

            memories_passages = []
            memory_counter = 0

            for m in mems_to_filter[start_idx:end_idx]:
                if self.dual_db_interface is None:
                    colour_print("RED", "Error: dual_db_interface is None")
                    continue
                interactions = self.dual_db_interface.get_interactions_from_memories([m], self.session.session_id)
                if interactions:
                    um, am = interactions[0]  # Get the first (and only) interaction
                    memories_passages.append(
                        f"{memory_counter}). (User): {um.passage}\n(You): {am.passage}\nKeywords: {m.metadata['keywords']}"
                    )
                    memory_counter += 1

            passages = "\n----\n".join(memories_passages)
            context = [
                make_user_message(
                    prompt.format(passages=passages, situation=situation))
            ]

            while True:
                try:
                    print("Attempting filter")
                    result = ask_llm(
                        context,
                        model=self.model,
                        max_overall_tokens=self.max_prompt_size,
                        cost_callback=cost_cb,
                        temperature=self.temperature,
                    )
                    # debug_actions(
                    #     context,
                    #     self.temperature,
                    #     result,
                    #     self.llm_call_idx,
                    #     self.debug_level,
                    #     self.save_name,
                    #     name_template="reply-{idx}-filter-" + str(call_count),
                    # )

                    json_list = sanitize_and_parse_json(result)
                    for idx, selected_object in enumerate(json_list):
                        if selected_object["related"]:
                            filtered_mems.append(mems_to_filter[start_idx +
                                                                idx])

                    call_count += 1
                    break
                except Exception as e:
                    print(e)
                    continue

        filtered_mems = sorted(filtered_mems, key=lambda x: x.timestamp)
        print("Memories after LLM filtering")
        for m in filtered_mems:
            colour_print("GREEN", m)

        return filtered_mems

    def get_relevant_memories(self, user_message: str, cost_cb: Callable) -> List[Memory]:
        colour_print("CYAN", f"Starting get_relevant_memories for message: {user_message}")
        
        query_dict = self._generate_queries(user_message, cost_cb)
        all_retrieved_memories = self._retrieve_memories(query_dict, user_message)
        
        relevance_filtered_mems = self._filter_by_relevance(all_retrieved_memories, query_dict["keywords"])
        keyword_filtered_mems = self._filter_by_keywords(relevance_filtered_mems, query_dict["keywords"])
        
        keyword_filtered_mems = self._perform_spreading_activations(keyword_filtered_mems)
        
        # # TODO: Uncomment all this stuff when doing dev stuff
        # trivia_skip = False
        # for kw in query_dict["keywords"]:
        #     if "trivia" in kw:
        #         trivia_skip = True
        #
        # if trivia_skip:
        #     llm_filtered_mems = keyword_filtered_mems
        # else:
        #     llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], query_dict["keywords"], cost_cb)

        # TODO: ....And comment this one out
        llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems,
                                                query_dict["queries"],
                                                query_dict["keywords"], cost_cb)
        colour_print("GREEN", f"Memories after LLM filtering: {len(llm_filtered_mems)}")
        
        sorted_mems = sorted(llm_filtered_mems, key=lambda x: x.timestamp)
        colour_print("CYAN", f"Returning {len(sorted_mems)} sorted memories")
        return sorted_mems

    def _generate_queries(self, user_message: str, cost_cb: Callable) -> Dict[str, List[str]]:
        prompt = """Message from user: "{user_message}"
        
        Given the above user question/statement, your task is to provide semantic queries and keywords for searching an archived 
        conversation history that may be relevant to a reply to the user.

        The search queries you produce should be compact reformulations of the user question/statement,
        taking context into account. The purpose of the queries is accurate information retrieval. 
        Search is purely semantic. 

        Create a general query and a specific query. Pay attention to the situation and topic of the conversation including any characters or specifically named persons.
        Use up to three of these keywords to help narrow the search:
        {keywords}

        The current time is: {time}. 

        Write JSON in the following format:

        {{
            "queries": array, // An array of strings: 2 descriptive search phrases, one general and one specific
            "keywords": array // An array of strings: 1 to 3 keywords that can be used to narrow the category of memories that are interesting. 
        }}"""

        context = [
            make_user_message(
                prompt.format(
                    user_message=user_message,
                    time=self.now,
                    keywords=self.dual_db_interface.get_all_keywords(),
                ))
        ]
        colour_print("YELLOW", f"Context created with prompt and keywords")
        
        while True:
            colour_print("YELLOW", "Generating queries")
            response = ask_llm(
                context,
                model=self.model,
                max_overall_tokens=self.max_prompt_size,
                cost_callback=cost_cb,
                temperature=self.temperature,
            )
            colour_print("GREEN", f"LLM response received: {response[:100]}...")  # Print first 100 chars

            try:
                query_dict = sanitize_and_parse_json(response)
                query_dict["keywords"] = [k.lower() for k in query_dict["keywords"]]
                colour_print("GREEN", f"Query keywords: {query_dict['keywords']}")
                return query_dict
            except Exception as e:
                colour_print("RED", f"Error occurred: {str(e)}")
                colour_print("YELLOW", "Retrying query generation")

    def _retrieve_memories(self, query_dict: Dict[str, List[str]], user_message: str) -> List[Memory]:
        all_retrieved_memories = []
        for q in query_dict["queries"] + [user_message]:
            colour_print("YELLOW", f"Querying with: {q}")
            memories = self.dual_db_interface.get_relevant_memories(
                q, self.session.session_id, limit=100)
            colour_print("GREEN", f"Retrieved {len(memories)} memories for query: {q}")
            all_retrieved_memories.extend(memories)
        
        colour_print("GREEN", f"Total retrieved memories: {len(all_retrieved_memories)}")
        return all_retrieved_memories

    def _filter_by_relevance(self, memories: List[Memory], keywords: List[str]) -> List[Memory]:
        relevance_filtered_mems = [
            x for x in memories if x.relevance > 0.6
        ] + self.dual_db_interface.retrieve_from_keywords(
            keywords, self.session.session_id)
        colour_print("GREEN", f"Memories after relevance filtering: {len(relevance_filtered_mems)}")
        return relevance_filtered_mems

    def _filter_by_keywords(self, memories: List[Memory], keywords: List[str]) -> List[Memory]:
        keyword_filtered_mems = [
            m for m in memories
            if any(kw in keywords for kw in m.metadata["keywords"])
        ]
        colour_print("GREEN", f"Memories after keyword filtering: {len(keyword_filtered_mems)}")
        return keyword_filtered_mems

    def _perform_spreading_activations(self, memories: List[Memory]) -> List[Memory]:
        colour_print("YELLOW", f"Performing spreading activations with {len(memories[:10])} memories.")
        secondary_memories = []
        for mem in memories[:10]:
            for r_mem in self.dual_db_interface.get_relevant_memories(
                    mem.passage, self.session.session_id, limit=5):
                if (r_mem.relevance > 0.6 and r_mem not in secondary_memories
                        and r_mem not in memories):
                    secondary_memories.append(r_mem)

        memories.extend(secondary_memories)
        colour_print("GREEN", f"Memories after spreading activations: {len(memories)}")
        return memories

    def save_interaction(self, user_message: str, response_message: str,
                         keywords: List[str]):
        self.session.add_interaction(
            (
                Message(role="user",
                        content=user_message,
                        timestamp=self.now.timestamp()),
                Message(
                    role="assistant",
                    content=response_message,
                    timestamp=self.now.timestamp(),
                ),
            ),
            keywords,
        )

    def reset(self):
        self.dual_db_interface.clear_all()
        self.new_session()

    def state_as_text(self) -> str:
        """
        :return: A string representation of the content of the agent's memories (including
        embeddings and chunks) in addition to agent configuration information.
        Note that callback functions are not part of the provided state string.
        """
        state = {
            "model": self.model,
            "max_prompt_size": self.max_prompt_size,
            "max_completion_tokens": self.max_completion_tokens,
            "session": self.session.state_as_text(),
        }
        return json.dumps(state, cls=SimpleJSONEncoder)

    def from_state_text(
        self,
        state_text: str,
        prompt_callback: Optional[Callable[[str, str, List[Dict], str],
                                           Any]] = None,
    ):
        """
        Builds an LTMAgent given a state string previously obtained by
        calling the state_as_text() method.
        :param state_text: A string previously obtained by calling the state_as_text() method.
        :param prompt_callback: Optional function used to get information on prompts sent to the LLM.
        :return:
        """
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        self.max_prompt_size = state["max_prompt_size"]
        self.max_completion_tokens = state["max_completion_tokens"]
        self.model = state["model"]
        self.prompt_callback = prompt_callback
        self.session = LTMAgentSession.from_state_text(state["session"],
                                                       self.dual_db_interface)

    def close(self):
        self.dual_db_interface.close()
