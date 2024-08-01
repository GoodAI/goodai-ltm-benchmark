import json
import uuid
from dataclasses import dataclass, field
import datetime
from math import ceil
from typing import Optional, Callable, Any, Dict
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Table, ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from contextlib import contextmanager

from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory
from litellm import token_counter
from model_interfaces.base_ltm_agent import Message
from utils.constants import DATA_DIR

from utils.llm import  make_system_message, make_user_message, ask_llm
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

class DBSession(Base):
    __tablename__ = 'sessions'
    session_id = Column(String, primary_key=True)
    messages = relationship("DBMessage", back_populates="session")
    keywords = relationship("Keyword", secondary="message_keywords", back_populates="messages")

class DBMessage(Base):
    __tablename__ = 'messages'
    timestamp = Column(Float, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.session_id'), primary_key=True)
    role = Column(String)
    content = Column(Text)
    session = relationship("DBSession", back_populates="messages")
    id = Column(Integer, primary_key=True, autoincrement=True)

class DBInteraction(Base):
    __tablename__ = 'interactions'
    timestamp = Column(Float, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.session_id'), primary_key=True)
    user_message_id = Column(Integer, ForeignKey('messages.id'))
    assistant_message_id = Column(Integer, ForeignKey('messages.id'))
    user_message = relationship("DBMessage", foreign_keys=[user_message_id])
    assistant_message = relationship("DBMessage", foreign_keys=[assistant_message_id])

class Keyword(Base):
    __tablename__ = 'keywords'
    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True)
    messages = relationship("DBMessage", secondary="message_keywords", back_populates="keywords")

message_keywords = Table('message_keywords', Base.metadata,
    Column('message_timestamp', Float, primary_key=True),
    Column('session_id', String, primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id'), primary_key=True),
    ForeignKeyConstraint(['message_timestamp', 'session_id'], ['messages.timestamp', 'messages.session_id'])
)

@dataclass
class Memory:
    passage: str
    metadata: Dict[str, Any]
    timestamp: float
    relevance: float = 1.0

class LTMAgentSession:
    """
    An agent session, or a collection of messages.
    """
    def __init__(self, session_id: str, db_manager: DatabaseManager):
        self.session_id = session_id
        self.db_manager = db_manager

        with self.db_manager.session_scope() as session:
            # Create a new database session entry
            db_session = DBSession(session_id=self.session_id)
            session.add(db_session)

    @property
    def message_count(self):
        with self.db_manager.session_scope() as session:
            return session.query(DBMessage).filter_by(session_id=self.session_id).count()

    def state_as_text(self) -> str:
        """
        :return: A string that represents the contents of the session.
        """
        with self.db_manager.session_scope() as session:
            messages = session.query(DBMessage).filter_by(session_id=self.session_id).order_by(DBMessage.timestamp).all()
            interactions = session.query(DBInteraction).filter_by(session_id=self.session_id).order_by(DBInteraction.timestamp).all()
            
            state = {
                "session_id": self.session_id,
                "history": [Message(role=m.role, content=m.content, timestamp=m.timestamp) for m in messages],
                "interactions": {
                    str(i.timestamp): (
                        Message(role=i.user_message.role, content=i.user_message.content, timestamp=i.user_message.timestamp),
                        Message(role=i.assistant_message.role, content=i.assistant_message.content, timestamp=i.assistant_message.timestamp)
                    ) for i in interactions
                }
            }
        
        return json.dumps(state, cls=SimpleJSONEncoder)

    def add_interaction(self, interaction: tuple[Message, Message], keywords: list[str]):
        with self.db_manager.session_scope() as session:
            user_message, assistant_message = interaction
            db_user_message = DBMessage(
                timestamp=user_message.timestamp,
                session_id=self.session_id,
                role=user_message.role,
                content=user_message.content
            )
            db_assistant_message = DBMessage(
                timestamp=assistant_message.timestamp,
                session_id=self.session_id,
                role=assistant_message.role,
                content=assistant_message.content
            )
            session.add(db_user_message)
            session.add(db_assistant_message)
            session.flush()  # This will assign ids to the messages
            
            db_interaction = DBInteraction(
                timestamp=user_message.timestamp,
                session_id=self.session_id,
                user_message_id=db_user_message.id,
                assistant_message_id=db_assistant_message.id
            )
            session.add(db_interaction)
            
            for kw in keywords:
                keyword = session.query(Keyword).filter_by(keyword=kw).first()
                if not keyword:
                    keyword = Keyword(keyword=kw)
                    session.add(keyword)
                db_user_message.keywords.append(keyword)
                db_assistant_message.keywords.append(keyword)

    def interaction_from_timestamp(self, timestamp: float) -> tuple[Message, Message]:
        with self.db_manager.session_scope() as session:
            db_interaction = session.query(DBInteraction).filter_by(
                timestamp=timestamp, session_id=self.session_id
            ).first()
            if db_interaction:
                user_message = Message(
                    role=db_interaction.user_message.role,
                    content=db_interaction.user_message.content,
                    timestamp=db_interaction.user_message.timestamp
                )
                assistant_message = Message(
                    role=db_interaction.assistant_message.role,
                    content=db_interaction.assistant_message.content,
                    timestamp=db_interaction.assistant_message.timestamp
                )
                return (user_message, assistant_message)
        return None

    def by_index(self, idx):
        with self.db_manager.session_scope() as session:
            message = session.query(DBMessage).filter_by(session_id=self.session_id).order_by(DBMessage.timestamp).offset(idx).first()
            if message:
                return Message(role=message.role, content=message.content, timestamp=message.timestamp)
        return None

    @classmethod
    def from_state_text(cls, state_text: str, db_manager: DatabaseManager) -> 'LTMAgentSession':
        """
        Builds a session object given state text.
        :param state_text: Text previously obtained using the state_as_text() method.
        :param db_manager: DatabaseManager instance for database operations.
        :return: A session instance.
        """
        state: dict = json.loads(state_text, cls=SimpleJSONDecoder)
        session_id = state["session_id"]
        
        session = cls(session_id, db_manager)
        
        with db_manager.session_scope() as db_session:
            for message in state["history"]:
                db_message = DBMessage(
                    timestamp=message.timestamp,
                    session_id=session_id,
                    role=message.role,
                    content=message.content
                )
                db_session.add(db_message)
            
            for timestamp, (user_msg, assistant_msg) in state["interactions"].items():
                db_interaction = DBInteraction(
                    timestamp=float(timestamp),
                    session_id=session_id,
                    user_message_id=user_msg.id,
                    assistant_message_id=assistant_msg.id
                )
                db_session.add(db_interaction)
        
        return session

    def get_interaction(self, timestamp: float) -> tuple[Message, Message]:
        return self.interaction_from_timestamp(timestamp)

@dataclass
class InsertedContextAgent:
    max_completion_tokens: Optional[int] = None
    semantic_memory: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    max_prompt_size: int = 16384
    is_local: bool = True
    defined_kws: list = field(default_factory=list) #! SHOULD BE FROM DB
    llm_call_idx: int = 0
    costs_usd: float = 0.0
    model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: float = 0.01
    system_message = """You are a helpful AI assistant."""
    debug_level: int = 1
    session: LTMAgentSession = None
    now: datetime.datetime = None  # Set in `reply` to keep a consistent "now" timestamp
    db_url: str = "sqlite:///ltm_sessions.db"

    def sync_semantic_memory(self):
        with self.db_manager.session_scope() as session:
            messages = session.query(DBMessage).order_by(DBMessage.timestamp).all()
            self.semantic_memory.clear()
            for message in messages:
                keywords = [kw.keyword for kw in message.keywords]
                self.semantic_memory.add_text(message.content, timestamp=message.timestamp, metadata={"keywords": keywords})
                self.semantic_memory.add_separator()

    @property
    def save_name(self) -> str:
        return f"{self.model}-{self.max_prompt_size}-{self.max_completion_tokens}__{self.init_timestamp}"

    def __post_init__(self):
        self.semantic_memory = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=50, chunk_overlap_fraction=0.0))
        self.max_message_size = 1000
        self.defined_kws = [] #! SHOULD BE FROM DB
        self.init_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Initialize database
        self.db_manager = DatabaseManager(self.engine)
        
        self.new_session()
        
    def new_session(self) -> 'LTMAgentSession':
        """
        Creates a new LTMAgentSession object and sets it as the current session.
        :return: The new session object.
        """
        session_id = str(uuid.uuid4())
        self.session = LTMAgentSession(session_id=session_id, m_history=[], i_dict={}, db_manager=self.db_manager)
        if not self.semantic_memory.is_empty():
            self.semantic_memory.add_separator()
        return self.session

    def reply(self, user_message: str, agent_response: Optional[str] = None, cost_callback: Callable=None) -> str:

        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")
        self.now = datetime.datetime.now()

        keywords = self.keywords_for_message(user_message, cost_cb=cost_callback)
        context = self.create_context(user_message, max_prompt_size=self.max_prompt_size, previous_interactions=0, cost_cb=cost_callback)
        response_text = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
        debug_actions(context, self.temperature, response_text, self.llm_call_idx, self.debug_level, self.save_name,  name_template="reply-{idx}")
        self.llm_call_idx += 1

        # Save interaction to memory
        um = Message(role="user", content=user_message, timestamp=self.now.timestamp())
        am = Message(role="assistant", content=response_text, timestamp=self.now.timestamp())

        self.save_interaction(user_message, response_text, keywords)
        self.session.add_interaction((um, am), keywords)

        return response_text

    def keywords_for_message(self, user_message, cost_cb):

        prompt = 'Create two keywords to describe the topic of this message:\n"{user_message}".\n\nFocus on the topic and tone of the message. Produce the keywords in JSON like: `["keyword_1", "keyword_2"]`\n\nChoose keywords that would aid in retriving this message from memory in the future.\n\nReuse these keywords if appropriate: {keywords}'

        context = [make_system_message(prompt.format(user_message=user_message, keywords=self.defined_kws))]
        while True:
            try:
                print("Keyword gen")
                response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)

                keywords = [k.lower() for k in sanitize_and_parse_json(response)]
                break
            except Exception as e:
                print(repr(e) + response)
                continue

        # Update known list of keywords
        for k in keywords:
            if k not in self.defined_kws:
                self.defined_kws.append(k)

        print(f"Interaction keywords: {keywords}")
        return keywords
    
    def get_message_keywords(self, timestamp: float, session_id: str) -> list[str]:
        message = self.db_session.query(DBMessage).filter_by(timestamp=timestamp, session_id=session_id).first() #! should be interactions
        return [kw.keyword for kw in message.keywords] if message else []

    def create_context(self, user_message, max_prompt_size, previous_interactions, cost_cb):

        stamped_user_message = str(self.now) + ": " + user_message
        context = [make_system_message(self.system_message), make_user_message(stamped_user_message)]
        relevant_memories = self.get_relevant_memories(user_message, cost_cb)

        # Get interactions from the memories
        with self.db_manager.session_scope() as session:
            relevant_interactions = session.query(DBInteraction).filter(
                DBInteraction.session_id == self.session.session_id,
                DBInteraction.timestamp.in_([m.timestamp for m in relevant_memories])
            ).all()

            full_interactions = []
            for interaction in relevant_interactions: #? NEEDS REVIEWED
                timestamp = interaction.user_message.timestamp
                user_message = Message(
                    # role=interaction.user_message.role,
                    content=interaction.user_message.content,
                    # timestamp=interaction.user_message.timestamp
                )
                assistant_message = Message(
                    # role=interaction.assistant_message.role,
                    content=interaction.assistant_message.content,
                    # timestamp=interaction.assistant_message.timestamp
                )
                full_interactions.append((timestamp, user_message, assistant_message))

        for m in full_interactions:
            if "trivia" in m[0].content:
                colour_print("YELLOW", f"<*** trivia ***>")
            else:
                colour_print("YELLOW", f"{m[0].content}")

        # Add the previous messages
        final_idx = self.session.message_count - 1
        while previous_interactions > 0 and final_idx > 0:

            # Agent reply
            context.insert(1, self.session.by_index(final_idx).as_llm_dict())
            # User message
            context.insert(1, self.session.by_index(final_idx-1).as_llm_dict())

            final_idx -= 2
            previous_interactions -= 1

        # Add in memories up to the max prompt size
        current_size = token_counter(self.model, messages=context)
        shown_mems = 0
        target_size = max_prompt_size - self.max_message_size

        for interaction in full_interactions[::-1]:
            user_message, assistant_message = interaction
            future_size = token_counter(self.model, messages=context + [user_message.as_llm_dict(), assistant_message.as_llm_dict()])

            # If this message is going to be too big, then skip it
            if shown_mems >= 100:
                break

            if future_size > target_size:
                continue

            # Add the interaction and count the tokens
            if user_message not in context:
                context.insert(1, assistant_message.as_llm_dict())

                ts = datetime.datetime.fromtimestamp(user_message.timestamp)
                et_descriptor = f"{str(ts)[:-7]} ({td_format(self.now - ts)}) "
                context.insert(1, user_message.as_llm_dict())
                context[1]["content"] = et_descriptor + context[1]["content"]

                shown_mems += 1

                current_size = future_size

        print(f"current context size: {current_size}")

        return context

    def llm_memory_filter(self, memories, queries, keywords, cost_cb):

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

        added_timestamps = []
        mems_to_filter = []  # Memories without duplicates
        filtered_mems = []

        # Get the situation
        queries_txt = "- " + "\n- ".join(queries)
        context = [make_user_message(situation_prompt.format(queries=queries_txt))]
        situation = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
        colour_print("MAGENTA", f"Filtering situation: {situation}")

        # Remove memories that map to the same interaction
        with self.db_manager.session_scope() as session:
            for m in mems_to_filter:
                interaction = session.query(DBInteraction).filter_by(
                    timestamp=m.timestamp,
                    session_id=self.session.session_id
                ).first()
                
                if interaction:
                    um = Message(
                        role=interaction.user_message.role,
                        content=interaction.user_message.content,
                        timestamp=interaction.user_message.timestamp
                    )
                    am = Message(
                        role=interaction.assistant_message.role,
                        content=interaction.assistant_message.content,
                        timestamp=interaction.assistant_message.timestamp
                    )
                    memories_passages.append(f"{memory_counter}). (User): {um.content}\n(You): {am.content}\nKeywords: {m.metadata['keywords']}")
                    memory_counter += 1

        num_splices = ceil(len(mems_to_filter) / splice_length)
        # Iterate through the mems_to_filter list and create the passage
        call_count = 0
        for splice in range(num_splices):
            start_idx = splice * splice_length
            end_idx = (splice + 1) * splice_length

            memories_passages = []
            memory_counter = 0

            for m in mems_to_filter[start_idx:end_idx]:
                timestamp = m.timestamp
                um, am = self.session.interaction_from_timestamp(timestamp)
                memories_passages.append(f"{memory_counter}). (User): {um.content}\n(You): {am.content}\nKeywords: {m.metadata['keywords']}")
                memory_counter += 1

            passages = "\n----\n".join(memories_passages)
            context = [make_user_message(prompt.format(passages=passages, situation=situation))]

            while True:
                try:
                    print("Attempting filter")
                    result = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                    debug_actions(context, self.temperature, result, self.llm_call_idx, self.debug_level, self.save_name, name_template="reply-{idx}-filter-" + str(call_count))

                    json_list = sanitize_and_parse_json(result)
                    for idx, selected_object in enumerate(json_list):
                        if selected_object["related"]:
                            filtered_mems.append(mems_to_filter[idx + start_idx])

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

    def get_relevant_memories(self, user_message, cost_cb):
        prompt ="""Message from user: "{user_message}"
        
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

        # Now create the context for generating the queries
        context = [make_user_message(prompt.format(user_message=user_message, time=self.now, keywords=self.defined_kws))]
        all_retrieved_memories = []
        query_keywords = []
        while True:
            print("generating queries")
            response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)

            try:
                query_dict = sanitize_and_parse_json(response)
                query_keywords = [k.lower() for k in query_dict["keywords"]]
                print(f"Query keywords: {query_keywords}")

                all_retrieved_memories = []
                for q in query_dict["queries"] + [user_message]:
                    print(f"Querying with: {q}")
                    for mem in self.semantic_memory.retrieve(q, k=100):
                        if not self.memory_present(mem, all_retrieved_memories):
                            all_retrieved_memories.append(mem)
                break
            except Exception:
                continue

        # Filter by both relevance and keywords
        all_keywords = query_keywords
        relevance_filtered_mems = [x for x in all_retrieved_memories if x.relevance > 0.6] + self.retrieve_from_keywords(all_keywords)
        keyword_filtered_mems = []

        for m in relevance_filtered_mems:
            for kw in m.metadata["keywords"]:
                if kw in all_keywords:
                    keyword_filtered_mems.append(m)
                    break

        keyword_filtered_mems.extend(self.retrieve_from_keywords(all_keywords))

        # Spreading activations
        print(f"Performing spreading activations with {len(keyword_filtered_mems[:10])} memories.")
        secondary_memories = []
        for mem in keyword_filtered_mems[:10]:
            # print(f"Spreading with: {mem.passage}")
            for r_mem in self.semantic_memory.retrieve(mem.passage, k=5):
                if r_mem.relevance > 0.6 and not self.memory_present(r_mem, secondary_memories) and not self.memory_present(r_mem, keyword_filtered_mems):
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
        llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], all_keywords, cost_cb)

        sorted_mems = sorted(llm_filtered_mems, key=lambda x: x.timestamp)
        return sorted_mems

    def memory_present(self, memory, memory_list):
        # passage_info seems to be unique to memory, regardless of the query
        for list_mem in memory_list:
            if memory.passage_info.fromIndex == list_mem.passage_info.fromIndex and memory.passage_info.toIndex == list_mem.passage_info.toIndex:
                return True
        return False

    def save_interaction(self, user_message, response_message, keywords):
        self.semantic_memory.add_text(user_message, timestamp=self.now.timestamp(), metadata={"keywords": keywords})
        self.semantic_memory.add_separator()
        self.semantic_memory.add_text(response_message, timestamp=self.now.timestamp(), metadata={"keywords": keywords})
        self.semantic_memory.add_separator()

    def retrieve_from_keywords(self, keywords):
        db_messages = self.session.db_session.query(DBMessage).join(DBMessage.keywords).filter(Keyword.keyword.in_(keywords)).all()
        
        # Convert DB messages to memory format
        selected_mems = []
        for db_message in db_messages:
            mem = self.db_message_to_memory(db_message)
            if mem not in selected_mems:
                selected_mems.append(mem)
        
        return selected_mems
    
    def db_message_to_memory(self, db_message):
        return Memory(
            passage=db_message.content,
            metadata={"keywords": [kw.keyword for kw in db_message.keywords]},
            timestamp=db_message.timestamp,
            relevance=1.0  # You might want to calculate this based on some criteria
        )

    def reset(self):
        self.semantic_memory.clear()
        self.new_session()

    def state_as_text(self) -> str:
        """
        :return: A string representation of the content of the agent's memories (including
        embeddings and chunks) in addition to agent configuration information.
        Note that callback functions are not part of the provided state string.
        """
        state = dict(
            model=self.model,
            max_prompt_size=self.max_prompt_size,
            max_completion_tokens=self.max_completion_tokens,
            convo_mem=self.semantic_memory.state_as_text(),
            session=self.session.state_as_text(),
        )
        return json.dumps(state, cls=SimpleJSONEncoder)

    def from_state_text(self, state_text: str, prompt_callback: Callable[[str, str, list[dict], str], Any] = None):
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
        self.semantic_memory.set_state(state["convo_mem"])
        self.session = LTMAgentSession.from_state_text(state["session"])

    def close(self):
        if self.session:
            self.session.close_db_session()
        if hasattr(self, 'SessionMaker'):
            self.SessionMaker.close_all()


def td_format(td: datetime.timedelta) -> str: #? Will be part of utils.text.py
    seconds = int(td.total_seconds())
    periods = [
        ('year', 3600*24*365), ('month', 3600*24*30), ('day', 3600*24), ('hour', 3600), ('minute', 60), ('second', 1)
    ]
    parts = list()
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            parts.append("%s %s%s" % (period_value, period_name, has_s))
    if len(parts) == 0:
        return "just now"
    if len(parts) == 1:
        return f"{parts[0]} ago"
    return " and ".join([", ".join(parts[:-1])] + parts[-1:]) + " ago"


def debug_actions(context: list[dict[str, str]], temperature: float, response_text: str, llm_call_idx: int, debug_level: int, save_name: str, name_template: str = None):
    if debug_level < 1:
        return

    # See if dir exists or create it, and set llm_call_idx
    save_dir = _debug_dir.joinpath(save_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    if llm_call_idx is None:
        if save_dir.exists() and len(list(save_dir.glob("*.txt"))) > 0:
            llm_call_idx = max(int(p.name.removesuffix(".txt")) for p in save_dir.glob("*.txt")) + 1
        else:
            llm_call_idx = 0

    # Write content of LLM call to file
    if name_template:
        save_path = save_dir.joinpath(f"{name_template.format(idx=llm_call_idx)}.txt")
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






