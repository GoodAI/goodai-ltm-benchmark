import json
import uuid
from dataclasses import dataclass, field, asdict
import datetime
from math import ceil
from typing import Optional, Callable, Any, Dict, List, Tuple
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

_debug_dir = DATA_DIR.joinpath("ltm_debug_info")

Base = declarative_base()


class DatabaseManager:

    def __init__(self, engine):
        self.SessionMaker = sessionmaker(bind=engine)

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
    messages = relationship("DBMessage",
                            secondary="message_keywords",
                            back_populates="keywords")


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

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: float,
        keywords: List[str],
    ) -> Memory:
        with self.db_manager.session_scope() as session:
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

            return memory

    def get_relevant_memories(self,
                              query: str,
                              session_id: Optional[str] = None,
                              limit: int = 10) -> List[Memory]:
        semantic_results = self.semantic_memory.retrieve(query, k=limit)

        with self.db_manager.session_scope() as session:
            memories = []
            for result in semantic_results:
                db_message = (session.query(DBMessage).filter_by(
                    semantic_id=result.id).first())
                if db_message and (session_id is None
                                   or db_message.session_id == session_id):
                    memory = Memory(
                        passage=db_message.content,
                        metadata={
                            "keywords":
                            [kw.keyword for kw in db_message.keywords],
                            "role": db_message.role,
                            "session_id": db_message.session_id,
                        },
                        timestamp=db_message.timestamp,
                        relevance=result.relevance,
                        semantic_id=result.id,
                        relational_id=db_message.id,
                    )
                    memories.append(memory)

            return memories

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
    model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: float = 0.01
    system_message: str = "You are a helpful AI assistant."
    debug_level: int = 1
    session: Optional[LTMAgentSession] = None
    now: Optional[datetime.datetime] = None
    db_url: str = "sqlite:///ltm_sessions.db"
    dual_db_interface: Optional[DualDatabaseInterface] = None

    def __post_init__(self):
        self.max_message_size = 1000
        self.init_timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S")

        # Initialize database
        engine = create_engine(self.db_url)
        db_manager = DatabaseManager(engine)
        semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(
            chunk_capacity=50, chunk_overlap_fraction=0.0))
        self.dual_db_interface = DualDatabaseInterface(db_manager,
                                                       semantic_memory)

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
        debug_actions(
            context,
            self.temperature,
            response_text,
            self.llm_call_idx,
            self.debug_level,
            self.save_name,
            name_template="reply-{idx}",
        )
        self.llm_call_idx += 1

        self.save_interaction(user_message, response_text, keywords)

        return response_text

    def keywords_for_message(self, user_message: str,
                             cost_cb: Callable) -> List[str]:
        defined_kws = self.dual_db_interface.get_all_keywords()
        prompt = f'Create two keywords to describe the topic of this message:\n"{user_message}".\n\nFocus on the topic and tone of the message. Produce the keywords in JSON like: `["keyword_1", "keyword_2"]`\n\nChoose keywords that would aid in retrieving this message from memory in the future.\n\nReuse these keywords if appropriate: {defined_kws}'

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
                keywords = [
                    k.lower() for k in sanitize_and_parse_json(response)
                ]
                break
            except Exception as e:
                print(repr(e) + response)
                continue

        self.dual_db_interface.add_keywords(keywords)
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

        for m in full_interactions:
            if "trivia" in m[0].passage:
                colour_print("YELLOW", f"<*** trivia ***>")
            else:
                colour_print("YELLOW", f"{m[0].passage}")

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
                interaction = self.dual_db_interface.get_interaction_from_memory(
                    m)
                if interaction:
                    um, am = interaction #! might be wrong
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
                    debug_actions(
                        context,
                        self.temperature,
                        result,
                        self.llm_call_idx,
                        self.debug_level,
                        self.save_name,
                        name_template="reply-{idx}-filter-" + str(call_count),
                    )

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
        # print("Memories after LLM filtering")
        # for m in filtered_mems:
        #     colour_print("GREEN", m)

        return filtered_mems

    def get_relevant_memories(self, user_message: str,
                              cost_cb: Callable) -> List[Memory]:
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
        while True:
            print("generating queries")
            response = ask_llm(
                context,
                model=self.model,
                max_overall_tokens=self.max_prompt_size,
                cost_callback=cost_cb,
                temperature=self.temperature,
            )

            try:
                query_dict = sanitize_and_parse_json(response)
                query_keywords = [k.lower() for k in query_dict["keywords"]]
                print(f"Query keywords: {query_keywords}")

                all_retrieved_memories = []
                for q in query_dict["queries"] + [user_message]:
                    print(f"Querying with: {q}")
                    all_retrieved_memories.extend(
                        self.dual_db_interface.get_relevant_memories(
                            q, self.session.session_id))
                break
            except Exception:
                continue

        # Filter by both relevance and keywords
        all_keywords = query_keywords
        relevance_filtered_mems = [
            x for x in all_retrieved_memories if x.relevance > 0.6
        ] + self.dual_db_interface.retrieve_from_keywords(
            all_keywords, self.session.session_id)
        keyword_filtered_mems = [
            m for m in relevance_filtered_mems
            if any(kw in all_keywords for kw in m.metadata["keywords"])
        ]

        # Spreading activations
        print(
            f"Performing spreading activations with {len(keyword_filtered_mems[:10])} memories."
        )
        secondary_memories = []
        for mem in keyword_filtered_mems[:10]:
            for r_mem in self.dual_db_interface.get_relevant_memories(
                    mem.passage, self.session.session_id, limit=5):
                if (r_mem.relevance > 0.6 and r_mem not in secondary_memories
                        and r_mem not in keyword_filtered_mems):
                    secondary_memories.append(r_mem)

        keyword_filtered_mems.extend(secondary_memories)

        # # TODO: Uncomment all this stuff when doing dev stuff
        # trivia_skip = False
        # for kw in all_keywords:
        #     if "trivia" in kw:
        #         trivia_skip = True
        #
        # if trivia_skip:
        #     llm_filtered_mems = keyword_filtered_mems
        # else:
        #     llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], all_keywords, cost_cb)

        # TODO: ....And comment this one out
        llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems,
                                                   query_dict["queries"],
                                                   all_keywords, cost_cb)

        sorted_mems = sorted(llm_filtered_mems, key=lambda x: x.timestamp)
        return sorted_mems

    def memory_present(self, memory, memory_list):
        # passage_info seems to be unique to memory, regardless of the query
        for list_mem in memory_list:
            if (memory.passage_info.fromIndex
                    == list_mem.passage_info.fromIndex
                    and memory.passage_info.toIndex
                    == list_mem.passage_info.toIndex):
                return True
        return False

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


def td_format(
        td: datetime.timedelta) -> str:  # ? Will be part of utils.text.py
    seconds = int(td.total_seconds())
    periods = [
        ("year", 3600 * 24 * 365),
        ("month", 3600 * 24 * 30),
        ("day", 3600 * 24),
        ("hour", 3600),
        ("minute", 60),
        ("second", 1),
    ]
    parts = list()
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = "s" if period_value > 1 else ""
            parts.append("%s %s%s" % (period_value, period_name, has_s))
    if len(parts) == 0:
        return "just now"
    if len(parts) == 1:
        return f"{parts[0]} ago"
    return " and ".join([", ".join(parts[:-1])] + parts[-1:]) + " ago"


def debug_actions(
    context: list[dict[str, str]],
    temperature: float,
    response_text: str,
    llm_call_idx: int,
    debug_level: int,
    save_name: str,
    name_template: str = None,
):
    if debug_level < 1:
        return

    # See if dir exists or create it, and set llm_call_idx
    save_dir = _debug_dir.joinpath(save_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    if llm_call_idx is None:
        if save_dir.exists() and len(list(save_dir.glob("*.txt"))) > 0:
            llm_call_idx = (max(
                int(p.name.removesuffix(".txt"))
                for p in save_dir.glob("*.txt")) + 1)
        else:
            llm_call_idx = 0

    # Write content of LLM call to file
    if name_template:
        save_path = save_dir.joinpath(
            f"{name_template.format(idx=llm_call_idx)}.txt")
    else:
        save_path = save_dir.joinpath(f"{llm_call_idx:06d}.txt")

    with open(save_path, "w") as fd:
        fd.write(f"LLM temperature: {temperature}\n")
        for m in context:
            fd.write(f"--- {m['role'].upper()}\n{m['content']}\n")
        fd.write(f"--- Response:\n{response_text}")

    # Wait for confirmation
    if debug_level < 2:
        return
    print(f"LLM call saved as {save_path.name}")
    input("Press ENTER to continue...")
