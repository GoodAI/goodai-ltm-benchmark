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
    Index,
)
from contextlib import contextmanager
from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from utils.constants import DATA_DIR
from utils.llm import make_system_message, make_user_message, ask_llm
from utils.ui import colour_print
from model_interfaces.base_ltm_agent import Message
import logging
from utils.text import td_format
from litellm import token_counter

logging.basicConfig(filename='ltm_agent.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
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
    semantic_id = Column(String, index=True)
    session = relationship("DBSession", back_populates="messages")
    keywords = relationship("Keyword", secondary="message_keywords", back_populates="messages")
    __table_args__ = (Index("idx_message_timestamp_session_role", "timestamp", "session_id", "role"), )

class DBInteraction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    user_message_id = Column(Integer, ForeignKey("messages.id"))
    assistant_message_id = Column(Integer, ForeignKey("messages.id"))
    user_message = relationship("DBMessage", foreign_keys=[user_message_id])
    assistant_message = relationship("DBMessage", foreign_keys=[assistant_message_id])
    __table_args__ = (Index("idx_interaction_timestamp_session", "timestamp", "session_id"), )

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
    relevance: float = 1.0
    semantic_id: Optional[str] = None
    relational_id: Optional[int] = None

class DualDatabaseInterface:
    def __init__(self, db_manager: DatabaseManager, semantic_memory):
        self.db_manager = db_manager
        self.semantic_memory = semantic_memory

    def add_message(self, session_id: str, role: str, content: str, timestamp: float, keywords: List[str]) -> Memory:
        with self.db_manager.session_scope() as session:
            db_message = DBMessage(timestamp=timestamp, session_id=session_id, role=role, content=content)
            session.add(db_message)
            session.flush()
            db_message.keywords.extend([self.db_manager.get_or_create_keyword(session, kw) for kw in keywords])
            semantic_id = self.semantic_memory.add_text(content, timestamp=timestamp,
                                                        metadata={"keywords": keywords, "role": role, "session_id": session_id})
            db_message.semantic_id = semantic_id
            return Memory(passage=content, metadata={"keywords": keywords, "role": role, "session_id": session_id},
                          timestamp=timestamp, semantic_id=semantic_id, relational_id=db_message.id)

    def get_relevant_memories(self, query: str, session_id: Optional[str] = None, limit: int = 10) -> List[Memory]:
        try:
            semantic_results = self.semantic_memory.retrieve(query, k=limit)
        except Exception as e:
            return self.retrieve_from_keywords(query.lower().split(), session_id, limit)
        with self.db_manager.session_scope() as session:
            memories = []
            for result in semantic_results:
                semantic_id = getattr(result, 'id', getattr(result, 'passage_id', None))
                if not semantic_id:
                    continue
                db_message = session.query(DBMessage).filter_by(semantic_id=semantic_id).first()
                if db_message and (session_id is None or db_message.session_id == session_id):
                    memories.append(self._create_memory_from_db_message(db_message, getattr(result, 'relevance', 1.0)))
            return memories

    def _create_memory_from_db_message(self, db_message: DBMessage, relevance: float = 1.0) -> Memory:
        return Memory(passage=db_message.content,
                      metadata={"keywords": [kw.keyword for kw in db_message.keywords], "role": db_message.role, "session_id": db_message.session_id},
                      timestamp=db_message.timestamp, relevance=relevance, semantic_id=db_message.semantic_id, relational_id=db_message.id)

    def retrieve_from_keywords(self, keywords: List[str], session_id: Optional[str] = None, limit: int = 10) -> List[Memory]:
        with self.db_manager.session_scope() as session:
            query = session.query(DBMessage).join(DBMessage.keywords)
            if session_id:
                query = query.filter(DBMessage.session_id == session_id)
            db_messages = query.filter(Keyword.keyword.in_(keywords)).limit(limit).all()
            return [self._create_memory_from_db_message(db_message) for db_message in db_messages]

    def get_all_keywords(self) -> List[str]:
        with self.db_manager.session_scope() as session:
            return [kw[0] for kw in session.query(Keyword.keyword).all()]

    def add_keywords(self, keywords: List[str]):
        with self.db_manager.session_scope() as session:
            existing_keywords = set(session.query(Keyword.keyword).filter(Keyword.keyword.in_(keywords)).all())
            new_keywords = set(keywords) - existing_keywords
            session.bulk_save_objects([Keyword(keyword=kw) for kw in new_keywords])

    def sync_databases(self):
        with self.db_manager.session_scope() as session:
            unsynced_messages = session.query(DBMessage).filter(DBMessage.semantic_id.is_(None)).all()
            for message in unsynced_messages:
                semantic_id = self.semantic_memory.add_text(message.content, timestamp=message.timestamp,
                                                            metadata={"keywords": [kw.keyword for kw in message.keywords], "role": message.role, "session_id": message.session_id})
                message.semantic_id = semantic_id

class LTMAgentSession:
    def __init__(self, session_id: str, dual_db_interface: DualDatabaseInterface):
        self.session_id = session_id
        self.dual_db_interface = dual_db_interface
        with self.dual_db_interface.db_manager.session_scope() as session:
            session.add(DBSession(session_id=self.session_id))

    @property
    def message_count(self):
        with self.dual_db_interface.db_manager.session_scope() as session:
            return session.query(DBMessage).filter_by(session_id=self.session_id).count()

    def state_as_text(self) -> str:
        with self.dual_db_interface.db_manager.session_scope() as session:
            messages = session.query(DBMessage).filter_by(session_id=self.session_id).order_by(DBMessage.timestamp).all()
            interactions = session.query(DBInteraction).filter_by(session_id=self.session_id).order_by(DBInteraction.timestamp).all()
            state = {
                "session_id": self.session_id,
                "history": [asdict(Memory(passage=m.content, metadata={"role": m.role, "keywords": [kw.keyword for kw in m.keywords]},
                                          timestamp=m.timestamp, semantic_id=m.semantic_id, relational_id=m.id)) for m in messages],
                "interactions": {str(i.timestamp): (asdict(Memory(passage=i.user_message.content, metadata={"role": i.user_message.role, "keywords": [kw.keyword for kw in i.user_message.keywords]},
                                                                timestamp=i.user_message.timestamp, semantic_id=i.user_message.semantic_id, relational_id=i.user_message.id)),
                                                 asdict(Memory(passage=i.assistant_message.content, metadata={"role": i.assistant_message.role, "keywords": [kw.keyword for kw in i.assistant_message.keywords]},
                                                                timestamp=i.assistant_message.timestamp, semantic_id=i.assistant_message.semantic_id, relational_id=i.assistant_message.id)))
                                 for i in interactions}
            }
        return json.dumps(state, cls=SimpleJSONEncoder)

    def add_interaction(self, interaction: Tuple[Message, Message], keywords: List[str]):
        user_message, assistant_message = interaction
        user_memory = self.dual_db_interface.add_message(self.session_id, user_message.role, user_message.content, user_message.timestamp, keywords)
        assistant_memory = self.dual_db_interface.add_message(self.session_id, assistant_message.role, assistant_message.content, assistant_message.timestamp, keywords)
        with self.dual_db_interface.db_manager.session_scope() as session:
            session.add(DBInteraction(timestamp=user_message.timestamp, session_id=self.session_id, user_message_id=user_memory.relational_id, assistant_message_id=assistant_memory.relational_id))

    def interaction_from_timestamp(self, timestamp: float) -> Optional[Tuple[Memory, Memory]]:
        with self.dual_db_interface.db_manager.session_scope() as session:
            db_interaction = session.query(DBInteraction).filter_by(timestamp=timestamp, session_id=self.session_id).first()
            if db_interaction:
                user_memory = self._create_memory_from_db_message(db_interaction.user_message)
                assistant_memory = self._create_memory_from_db_message(db_interaction.assistant_message)
                return (user_memory, assistant_memory)
        return None

    def _create_memory_from_db_message(self, db_message: DBMessage, relevance: float = 1.0) -> Memory:
        return Memory(passage=db_message.content, metadata={"role": db_message.role, "keywords": [kw.keyword for kw in db_message.keywords], "session_id": db_message.session_id},
                      timestamp=db_message.timestamp, relevance=relevance, semantic_id=db_message.semantic_id, relational_id=db_message.id)

    @classmethod
    def from_state_text(cls, state_text: str, dual_db_interface: DualDatabaseInterface) -> "LTMAgentSession":
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        session_id = state["session_id"]
        session = cls(session_id, dual_db_interface)
        for memory_dict in state["history"]:
            memory = Memory(**memory_dict)
            session.dual_db_interface.add_message(session_id=session_id, role=memory.metadata["role"], content=memory.passage, timestamp=memory.timestamp, keywords=memory.metadata["keywords"])
        for timestamp, (user_memory_dict, assistant_memory_dict) in state["interactions"].items():
            user_memory = Memory(**user_memory_dict)
            assistant_memory = Memory(**assistant_memory_dict)
            session.add_interaction((Message(role=user_memory.metadata["role"], content=user_memory.passage, timestamp=user_memory.timestamp),
                                     Message(role=assistant_memory.metadata["role"], content=assistant_memory.passage, timestamp=assistant_memory.timestamp)),
                                    keywords=user_memory.metadata["keywords"])
        return session

@dataclass
class InsertedContextAgent:
    max_completion_tokens: Optional[int] = None
    max_prompt_size: int = 16384
    is_local: bool = True
    llm_call_idx: int = 0
    costs_usd: float = 0.0
    model: str = "gpt-4o-mini"
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
        self.init_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if not self.dual_db_interface:
            engine = create_engine(self.db_url)
            db_manager = DatabaseManager(engine)
            db_manager.create_tables()
            semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=50, chunk_overlap_fraction=0.0))
            self.dual_db_interface = DualDatabaseInterface(db_manager, semantic_memory)
        self.new_session()

    @property
    def save_name(self) -> str:
        return f"{self.model}-{self.max_prompt_size}-{self.max_completion_tokens}__{self.init_timestamp}"

    def new_session(self) -> LTMAgentSession:
        session_id = str(uuid.uuid4())
        self.session = LTMAgentSession(session_id=session_id, dual_db_interface=self.dual_db_interface)
        return self.session

    def reply(self, user_message: str, agent_response: Optional[str] = None, cost_callback: Callable = None) -> str:
        self.now = datetime.datetime.now()
        keywords = self.keywords_for_message(user_message, cost_callback)
        context = self.create_context(user_message, max_prompt_size=self.max_prompt_size, previous_interactions=0, cost_cb=cost_callback)
        response_text = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
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
                response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                keywords = [k.lower() for k in sanitize_and_parse_json(response)]
                break
            except Exception as e:
                continue
        new_keywords = [kw for kw in keywords if kw not in defined_kws]
        if new_keywords:
            self.dual_db_interface.add_keywords(new_keywords)
        return keywords

    def create_context(self, user_message: str, max_prompt_size: int, previous_interactions: int, cost_cb: Callable) -> List[Dict[str, str]]:
        stamped_user_message = f"{self.now}: {user_message}"
        context = [make_system_message(self.system_message), make_user_message(stamped_user_message)]
        relevant_memories = self.get_relevant_memories(user_message, cost_cb)
        full_interactions = self.dual_db_interface.get_interactions_from_memories(relevant_memories, self.session.session_id)
        for user_memory, assistant_memory in full_interactions:
            colour_print("YELLOW", f"{user_memory.passage}")
        final_idx = self.session.message_count - 1
        while previous_interactions > 0 and final_idx > 0:
            context.insert(1, self.memory_to_dict(self.session.by_index(final_idx)))
            context.insert(1, self.memory_to_dict(self.session.by_index(final_idx - 1)))
            final_idx -= 2
            previous_interactions -= 1
        current_size = token_counter(self.model, messages=context)
        shown_mems = 0
        target_size = max_prompt_size - self.max_message_size
        for user_memory, assistant_memory in full_interactions[::-1]:
            future_size = token_counter(self.model, messages=context + [self.memory_to_dict(user_memory), self.memory_to_dict(assistant_memory)])
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
        return context

    def memory_to_dict(self, memory: Memory) -> Dict[str, str]:
        return {"role": memory.metadata["role"], "content": memory.passage}

    def get_relevant_memories(self, user_message: str, cost_cb: Callable) -> List[Memory]:
        query_dict = self._generate_queries(user_message, cost_cb)
        all_retrieved_memories = self._retrieve_memories(query_dict, user_message)
        relevance_filtered_mems = self._filter_by_relevance(all_retrieved_memories, query_dict["keywords"])
        keyword_filtered_mems = self._filter_by_keywords(relevance_filtered_mems, query_dict["keywords"])
        keyword_filtered_mems = self._perform_spreading_activations(keyword_filtered_mems)
        llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], query_dict["keywords"], cost_cb)
        return sorted(llm_filtered_mems, key=lambda x: x.timestamp)

    def _generate_queries(self, user_message: str, cost_cb: Callable) -> Dict[str, List[str]]:
        prompt = f'''Message from user: "{user_message}"
        
        Given the above user question/statement, your task is to provide semantic queries and keywords for searching an archived 
        conversation history that may be relevant to a reply to the user.

        The search queries you produce should be compact reformulations of the user question/statement,
        taking context into account. The purpose of the queries is accurate information retrieval. 
        Search is purely semantic. 

        Create a general query and a specific query. Pay attention to the situation and topic of the conversation including any characters or specifically named persons.
        Use up to three of these keywords to help narrow the search:
        {", ".join(self.dual_db_interface.get_all_keywords())}

        The current time is: {self.now}. 

        Write JSON in the following format:

        {{
            "queries": array, // An array of strings: 2 descriptive search phrases, one general and one specific
            "keywords": array // An array of strings: 1 to 3 keywords that can be used to narrow the category of memories that are interesting. 
        }}'''
        context = [make_user_message(prompt)]
        while True:
            try:
                response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                query_dict = sanitize_and_parse_json(response)
                query_dict["keywords"] = [k.lower() for k in query_dict["keywords"]]
                return query_dict
            except Exception as e:
                continue

    def _retrieve_memories(self, query_dict: Dict[str, List[str]], user_message: str) -> List[Memory]:
        all_retrieved_memories = []
        for q in query_dict["queries"] + [user_message]:
            memories = self.dual_db_interface.get_relevant_memories(q, self.session.session_id, limit=100)
            all_retrieved_memories.extend(memories)
        return all_retrieved_memories

    def _filter_by_relevance(self, memories: List[Memory], keywords: List[str]) -> List[Memory]:
        return [x for x in memories if x.relevance > 0.6] + self.dual_db_interface.retrieve_from_keywords(keywords, self.session.session_id)

    def _filter_by_keywords(self, memories: List[Memory], keywords: List[str]) -> List[Memory]:
        return [m for m in memories if any(kw in keywords for kw in m.metadata["keywords"])]

    def _perform_spreading_activations(self, memories: List[Memory]) -> List[Memory]:
        secondary_memories = []
        for mem in memories[:10]:
            secondary_memories.extend([r_mem for r_mem in self.dual_db_interface.get_relevant_memories(mem.passage, self.session.session_id, limit=5)
                                       if r_mem.relevance > 0.6 and r_mem not in secondary_memories and r_mem not in memories])
        memories.extend(secondary_memories)
        return memories

    def llm_memory_filter(self, memories: List[Memory], queries: List[str], keywords: List[str], cost_cb: Callable) -> List[Memory]:
        if not memories:
            return []
        prompt = f'''Here are a number of interactions, each is given a number:
{{passages}}         
*********
Each of these interactions might be related to the general situation below. Your task is to judge if these interaction have any relation to the situation.
Filter out interactions that very clearly do not have any relation. But keep in interactions that have any kind of relationship to the situation such as in: topic, characters, locations, setting, etc.
SITUATION:
{{situation}}
Express your answer in this JSON: 
[
    {{
        "number": int,
        "justification": string,
        "related": bool
    }},
    ...
]'''
        queries_txt = "- " + "\n- ".join(queries)
        situation_prompt = f'''You are a part of an agent. Another part of the agent is currently searching for memories using the statements below.
Based on these statements, describe what is currently happening external to the agent in general terms:
{queries_txt}'''
        context = [make_user_message(situation_prompt)]
        situation = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
        splice_length = 10
        mems_to_filter = list({m.timestamp: m for m in memories}.values())
        num_splices = ceil(len(mems_to_filter) / splice_length)
        filtered_mems = []
        for splice in range(num_splices):
            start_idx = splice * splice_length
            end_idx = (splice + 1) * splice_length
            passages = "\n----\n".join([f"{idx}). (User): {interaction[0].passage}\n(You): {interaction[1].passage}\nKeywords: {m.metadata['keywords']}"
                                        for idx, (m, interaction) in enumerate(zip(mems_to_filter[start_idx:end_idx], [self.dual_db_interface.get_interactions_from_memories([m], self.session.session_id)[0] for m in mems_to_filter[start_idx:end_idx]]))])
            context = [make_user_message(prompt.format(passages=passages, situation=situation))]
            while True:
                try:
                    result = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                    json_list = sanitize_and_parse_json(result)
                    filtered_mems.extend([mems_to_filter[start_idx + idx] for idx, selected_object in enumerate(json_list) if selected_object["related"]])
                    break
                except Exception as e:
                    continue
        return sorted(filtered_mems, key=lambda x: x.timestamp)

    def save_interaction(self, user_message: str, response_message: str, keywords: List[str]):
        self.session.add_interaction((Message(role="user", content=user_message, timestamp=self.now.timestamp()), Message(role="assistant", content=response_message, timestamp=self.now.timestamp())), keywords)

    def reset(self):
        self.dual_db_interface.clear_all()
        self.new_session()

    def state_as_text(self) -> str:
        return json.dumps({"model": self.model, "max_prompt_size": self.max_prompt_size, "max_completion_tokens": self.max_completion_tokens, "session": self.session.state_as_text()}, cls=SimpleJSONEncoder)

    def from_state_text(self, state_text: str, prompt_callback: Optional[Callable[[str, str, List[Dict], str], Any]] = None):
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        self.max_prompt_size = state["max_prompt_size"]
        self.max_completion_tokens = state["max_completion_tokens"]
        self.model = state["model"]
        self.prompt_callback = prompt_callback
        self.session = LTMAgentSession.from_state_text(state["session"], self.dual_db_interface)

    def close(self):
        self.dual_db_interface.close()
