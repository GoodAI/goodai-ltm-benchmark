import json
import re
import uuid
from dataclasses import dataclass, field
import datetime
from math import ceil
from typing import Optional, Callable, Any, List, Tuple, Dict, Union
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload

from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.base import RetrievedMemory
from litellm import token_counter
from model_interfaces.base_ltm_agent import Message

from utils.llm import make_system_message, make_user_message, ask_llm, log_llm_call
from utils.text import td_format
from utils.ui import colour_print

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ltm_agent.log'
)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

message_keywords = Table(
    'message_keywords', Base.metadata,
    Column('message_id', Integer, ForeignKey('messages.id'), primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id'), primary_key=True)
)

class DBMessage(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    timestamp = Column(Float, index=True)
    role = Column(String)
    content = Column(String)
    session_id = Column(String, index=True)
    keywords = relationship("DBKeyword", secondary=message_keywords, back_populates="messages")

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "keywords": [kw.keyword for kw in self.keywords] if 'keywords' in self.__dict__ else []
        }

class DBKeyword(Base):
    __tablename__ = 'keywords'
    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True, index=True)
    messages = relationship("DBMessage", secondary=message_keywords, back_populates="keywords")

class DatabaseManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    def add_message(self, message: Message, keywords: List[str], session_id: str):
        with self.session_scope() as session:
            db_message = DBMessage(
                timestamp=message.timestamp,
                role=message.role,
                content=message.content,
                session_id=session_id
            )
            for kw in keywords:
                keyword = session.query(DBKeyword).filter_by(keyword=kw).first()
                if not keyword:
                    keyword = DBKeyword(keyword=kw)
                db_message.keywords.append(keyword)
            session.add(db_message)

    def get_messages(self, limit: Optional[int] = None):
        messages = self.dual_memory.db_manager.get_messages(self.session_id, limit)
        return [(
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp},
            {"role": "assistant", "content": "", "timestamp": msg.timestamp}
        ) for msg in messages if msg.role == "user"]

    def get_keywords(self):
        with self.session_scope() as session:
            return [kw.keyword for kw in session.query(DBKeyword).all()]

    def get_messages_by_keywords(self, keywords: List[str], session_id: Optional[str] = None, limit: Optional[int] = None):
        with self.session_scope() as session:
            query = session.query(DBMessage).options(joinedload(DBMessage.keywords)).join(DBMessage.keywords).filter(DBKeyword.keyword.in_(keywords))
            if session_id:
                query = query.filter(DBMessage.session_id == session_id)
            if limit:
                query = query.limit(limit)
            return query.all()
        
    def convert_to_db_message(self, message: Union[Message, Dict], session_id: str) -> DBMessage:
        if isinstance(message, Message):
            return DBMessage(
                timestamp=message.timestamp,
                role=message.role,
                content=message.content,
                session_id=session_id
            )
        elif isinstance(message, dict):
            return DBMessage(
                timestamp=message['timestamp'],
                role=message['role'],
                content=message['content'],
                session_id=session_id
            )
        else:
            raise ValueError("Input must be either a Message object or a dictionary")

class DualMemorySystem:
    def __init__(self, db_manager: DatabaseManager, semantic_config: TextMemoryConfig):
        self.db_manager = db_manager
        self.semantic_memory = AutoTextMemory.create(config=semantic_config)

    def add_interaction(self, user_message: Message, assistant_message: Message, keywords: List[str], session_id: str):
        # Add to relational DB
        self.db_manager.add_message(user_message, keywords, session_id)
        self.db_manager.add_message(assistant_message, keywords, session_id)

        # Add to semantic memory (prioritizing user messages)
        # Comment: We're only storing user messages in semantic memory to prioritize user information
        self.semantic_memory.add_text(user_message.content, timestamp=user_message.timestamp, 
                                      metadata={"role": "user", "keywords": keywords, "session_id": session_id})
        self.semantic_memory.add_separator()

    def get_relevant_memories(self, query: str, session_id: str, k: int = 10) -> List[Tuple[DBMessage, DBMessage]]:
        semantic_results = self.semantic_memory.retrieve(query, k=k)
        return self._get_db_messages_from_semantic_results(semantic_results, session_id)

    def _get_db_messages_from_semantic_results(self, semantic_results: List[RetrievedMemory], session_id: str) -> List[Tuple[DBMessage, DBMessage]]:
        db_messages = []
        with self.db_manager.session_scope() as session:
            for result in semantic_results:
                user_message = session.query(DBMessage).filter(
                    DBMessage.timestamp == result.timestamp,
                    DBMessage.role == "user",
                    DBMessage.session_id == session_id
                ).first()
                if user_message:
                    assistant_message = session.query(DBMessage).filter(
                        DBMessage.session_id == user_message.session_id,
                        DBMessage.timestamp > user_message.timestamp,
                        DBMessage.role == "assistant"
                    ).order_by(DBMessage.timestamp).first()
                    if assistant_message:
                        db_messages.append((user_message, assistant_message))
                else:
                    # Fallback: string match on user message content if timestamp doesn't match
                    user_message = session.query(DBMessage).filter(
                        DBMessage.content == result.passage,
                        DBMessage.role == "user",
                        DBMessage.session_id == session_id
                    ).first()
                    if user_message:
                        assistant_message = session.query(DBMessage).filter(
                            DBMessage.session_id == user_message.session_id,
                            DBMessage.timestamp > user_message.timestamp,
                            DBMessage.role == "assistant"
                        ).order_by(DBMessage.timestamp).first()
                        if assistant_message:
                            db_messages.append((user_message, assistant_message))
        return db_messages

class LTMAgentSession:
    def __init__(self, session_id: str, dual_memory: DualMemorySystem):
        self.session_id = session_id
        self.dual_memory = dual_memory

    @property
    def message_count(self):
        return len(self.dual_memory.db_manager.get_messages(self.session_id))

    def add_interaction(self, interaction: Tuple[Message, Message], keywords: List[str]):
        user_message, assistant_message = interaction
        self.dual_memory.add_interaction(user_message, assistant_message, keywords, self.session_id)

    def get_messages(self, limit: Optional[int] = None):
        messages = self.dual_memory.db_manager.get_messages(self.session_id, limit)
        return [(Message(role=msg.role, content=msg.content, timestamp=msg.timestamp),
                Message(role=msg.role, content=msg.content, timestamp=msg.timestamp))
                for msg in messages if msg.role == "user"]

    def by_index(self, idx):
        messages = self.get_messages()
        return messages[idx] if idx < len(messages) else None

    def state_as_text(self) -> str:
        with self.dual_memory.db_manager.session_scope() as session:
            messages = self.dual_memory.db_manager.get_messages(self.session_id)
            state = {
                "session_id": self.session_id,
                "messages": [session.merge(msg).to_dict() for msg in messages]
            }
        return json.dumps(state, cls=SimpleJSONEncoder)

    @classmethod
    def from_state_text(cls, state_text: str, dual_memory: DualMemorySystem) -> 'LTMAgentSession':
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        session = cls(state["session_id"], dual_memory)
        for msg_data in state["messages"]:
            message = Message(role=msg_data["role"], content=msg_data["content"], timestamp=msg_data["timestamp"])
            session.dual_memory.db_manager.add_message(message, msg_data["keywords"], session.session_id)
            if msg_data["role"] == "user":
                session.dual_memory.semantic_memory.add_text(
                    message.content, 
                    timestamp=message.timestamp, 
                    metadata={"role": "user", "keywords": msg_data["keywords"], "session_id": session.session_id}
                )
        return session
    
    def _convert_to_db_message(self, message: Message) -> DBMessage:
        return DBMessage(
            timestamp=message.timestamp,
            role=message.role,
            content=message.content,
            session_id=self.session_id
        )

@dataclass
class InsertedContextAgent:
    max_completion_tokens: Optional[int] = None
    max_prompt_size: int = 16384
    is_local: bool = True
    llm_call_idx: int = 0
    model: str = "gpt-4"
    temperature: float = 0.01
    system_message: str = "You are a helpful AI assistant."
    debug_level: int = 1
    session: Optional[LTMAgentSession] = None
    now: Optional[datetime.datetime] = None
    db_url: str = "sqlite:///ltm_sessions.db"
    run_name: str = ""
    num_tries: int = 5

    def __post_init__(self):
        self.max_message_size = 1000
        db_manager = DatabaseManager(self.db_url)
        semantic_config = TextMemoryConfig(chunk_capacity=50, chunk_overlap_fraction=0.0)
        self.dual_memory = DualMemorySystem(db_manager, semantic_config)
        self.new_session()
        self.init_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    @property
    def save_name(self) -> str:
        return sanitize_filename(f"{self.model.replace('/', '-')}-{self.max_prompt_size}-{self.max_completion_tokens}__{self.init_timestamp}")

    def new_session(self) -> 'LTMAgentSession':
        session_id = str(uuid.uuid4())
        self.session = LTMAgentSession(session_id=session_id, dual_memory=self.dual_memory)
        return self.session

    def reply(self, user_message: str, agent_response: Optional[str] = None, cost_callback: Callable = None) -> str:
        logger.info(f"Received user message: {user_message}")
        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")
        self.now = datetime.datetime.now()

        keywords = self.keywords_for_message(user_message, cost_cb=cost_callback)
        context = self.create_context(user_message, max_prompt_size=self.max_prompt_size, previous_interactions=0, cost_cb=cost_callback)
        response_text = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_callback, temperature=self.temperature)
        log_llm_call(self.run_name, self.save_name, self.debug_level, label=f"reply-{self.llm_call_idx}")
        self.llm_call_idx += 1

        self.save_interaction(user_message, response_text, keywords)

        logger.info(f"Agent response: {response_text}")
        return response_text

    def keywords_for_message(self, user_message: str, cost_cb: Callable) -> List[str]:
        prompt = '''Create two keywords to describe the topic of this message:
"{user_message}".

Focus on the topic and tone of the message. Produce the keywords in JSON like: `["keyword_1", "keyword_2"]`

Choose keywords that would aid in retrieving this message from memory in the future.

Reuse these keywords if appropriate: {keywords}'''

        context = [make_system_message(prompt.format(user_message=user_message, keywords=self.dual_memory.db_manager.get_keywords()))]
        for _ in range(self.num_tries):
            try:
                logger.debug("Generating keywords")
                response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                keywords = [k.lower() for k in sanitize_and_parse_json(response)]
                break
            except Exception as e:
                logger.error(f"Error in keyword generation: {str(e)}")
                continue

        logger.info(f"Generated keywords: {keywords}")
        colour_print("YELLOW", f"Interaction keywords: {keywords}")
        return keywords

    def create_context(self, user_message: str, max_prompt_size: int, previous_interactions: int, cost_cb: Callable) -> List[Dict[str, str]]:
        with self.dual_memory.db_manager.session_scope() as session:
            context = [make_system_message(self.system_message), make_user_message(f"{str(self.now)[:-7]} ({td_format(datetime.timedelta(seconds=1))}) {user_message}")]
            relevant_interactions = self.get_relevant_memories(user_message, cost_cb)

            for user_interaction, assistant_interaction in relevant_interactions:
                if isinstance(user_interaction, dict):
                    content = user_interaction['content']
                else:  # Message object
                    content = user_interaction.content

                if "trivia" in content:
                    colour_print("YELLOW", "<*** trivia ***>")
                else:
                    colour_print("YELLOW", f"{content}")

            previous_messages = self.session.get_messages(previous_interactions)
            relevant_interactions.extend(previous_messages)

            current_size = token_counter(self.model, messages=context)
            shown_mems = 0
            target_size = max_prompt_size - self.max_message_size

            for user_interaction, assistant_interaction in reversed(relevant_interactions):
                user_dict = self._to_dict(user_interaction)
                assistant_dict = self._to_dict(assistant_interaction)
                
                future_size = token_counter(self.model, messages=context + [user_dict, assistant_dict])

                if shown_mems >= 100 or future_size > target_size:
                    break

                context.insert(1, assistant_dict)

                ts = datetime.datetime.fromtimestamp(user_dict['timestamp'])
                et_descriptor = f"{str(ts)[:-7]} ({td_format(self.now - ts)}) "
                user_dict["content"] = et_descriptor + user_dict["content"]
                context.insert(1, user_dict)

                shown_mems += 1
                current_size = future_size

            logger.debug(f"Current context size: {current_size}")
            return context

    def _to_dict(self, obj: Union[Dict, Message]) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, Message):
            return {
                "role": obj.role,
                "content": obj.content,
                "timestamp": obj.timestamp
            }
        else:
            raise ValueError(f"Unexpected object type: {type(obj)}")

    def llm_memory_filter(self, memories: List[Tuple[DBMessage, DBMessage]], queries: List[str], cost_cb: Callable) -> List[Tuple[DBMessage, DBMessage]]:
        situation_prompt = """You are a part of an agent. Another part of the agent is currently searching for memories using the statements below.
Based on these statements, describe what is currently happening external to the agent in general terms:
{queries}  
"""

        prompt = """Here are a number of interactions, each is given a number:
{passages}         
*****************************************

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

        if not memories:
            return []

        splice_length = 10
        filtered_interactions = []

        queries_txt = "- " + "\n- ".join(queries)
        context = [make_user_message(situation_prompt.format(queries=queries_txt))]
        situation = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
        colour_print("MAGENTA", f"Filtering situation: {situation}")
        logger.info(f"Filtering situation: {situation}")

        num_splices = ceil(len(memories) / splice_length)
        call_count = 0
        for splice in range(num_splices):
            start_idx = splice * splice_length
            end_idx = (splice + 1) * splice_length

            memories_passages = []
            for idx, mem in enumerate(memories[start_idx:end_idx], start=start_idx):
                memories_passages.append(f"[MEMORY NUMBER {idx} START].\n (User): {mem.passage}\nKeywords: {mem.metadata.get('keywords', [])}\n[MEMORY NUMBER {idx} END]")

            passages = "\n\n------------------------\n\n".join(memories_passages)
            context = [make_user_message(prompt.format(passages=passages, situation=situation))]

            for _ in range(self.num_tries):
                try:
                    logger.debug("Attempting memory filter")
                    result = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
                    log_llm_call(self.run_name, self.save_name, self.debug_level, label=f"reply-{self.llm_call_idx}-filter-{call_count}")

                    json_list = sanitize_and_parse_json(result)
                    for selected_object in json_list:
                        if selected_object["related"]:
                            filtered_interactions.append(memories[selected_object["number"]])

                    call_count += 1
                    break
                except Exception as e:
                    logger.error(f"Error in memory filtering: {str(e)}")
                    continue

        logger.info(f"Memory filtering complete. {len(filtered_interactions)} interactions selected.")
        return filtered_interactions

    def get_relevant_memories(self, user_message: str, cost_cb: Callable) -> List[Tuple[Dict, Dict]]:
        logger.info(f"Starting get_relevant_memories for message: {user_message}")
        colour_print("CYAN", f"Starting get_relevant_memories for message: {user_message}")
        
        query_dict = self._generate_queries(user_message, cost_cb)
        all_retrieved_memories = []
        
        for query in query_dict["queries"] + [user_message]:
            retrieved = self.dual_memory.semantic_memory.retrieve(query, k=100)
            all_retrieved_memories.extend(retrieved)
        
        relevance_filtered_mems = self._filter_by_relevance(all_retrieved_memories, " ".join(query_dict["keywords"]))
        keyword_filtered_mems = self._filter_by_keywords(relevance_filtered_mems, query_dict["keywords"])
        
        keyword_filtered_mems = self._perform_spreading_activations(keyword_filtered_mems)
        
        llm_filtered_mems = self.llm_memory_filter(keyword_filtered_mems, query_dict["queries"], cost_cb)
        
        sorted_mems = sorted(llm_filtered_mems, key=lambda x: x.timestamp)
        
        # Convert RetrievedMemory objects to dictionary pairs
        db_messages = []
        for mem in sorted_mems:
            user_msg = {
                'role': 'user',
                'content': mem.passage,
                'timestamp': mem.timestamp,
                'keywords': mem.metadata.get('keywords', [])
            }
            assistant_msg = {
                'role': 'assistant',
                'content': '',  # We don't have assistant content in RetrievedMemory
                'timestamp': mem.timestamp,
                'keywords': mem.metadata.get('keywords', [])
            }
            db_messages.append((user_msg, assistant_msg))

        colour_print("GREEN", f"Memories after filtering: {len(db_messages)}")
        logger.info(f"Memories after filtering: {len(db_messages)}")
        
        return db_messages

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

        context = [make_user_message(prompt.format(user_message=user_message, time=self.now, keywords=self.dual_memory.db_manager.get_keywords()))]
        logger.debug(f"Context created with prompt and keywords")
        colour_print("YELLOW", f"Context created with prompt and keywords")
        
        for _ in range(self.num_tries):
            logger.debug("Generating queries")
            colour_print("YELLOW", "Generating queries")
            response = ask_llm(context, model=self.model, max_overall_tokens=self.max_prompt_size, cost_callback=cost_cb, temperature=self.temperature)
            logger.debug(f"LLM response received: {response[:100]}...")
            colour_print("GREEN", f"LLM response received: {response[:100]}...")  # Print first 100 chars

            try:
                query_dict = sanitize_and_parse_json(response)
                query_dict["keywords"] = [k.lower() for k in query_dict["keywords"]]
                logger.info(f"Query keywords: {query_dict['keywords']}")
                colour_print("GREEN", f"Query keywords: {query_dict['keywords']}")
                return query_dict
            except Exception as e:
                logger.error(f"Error occurred: {str(e)}")
                colour_print("RED", f"Error occurred: {str(e)}")
                colour_print("YELLOW", "Retrying query generation")

    def _retrieve_memories(self, query_dict: Dict[str, List[str]], user_message: str) -> List[Tuple[DBMessage, DBMessage]]:
        all_retrieved_memories = []
        for q in query_dict["queries"] + [user_message]:
            logger.debug(f"Querying with: {q}")
            colour_print("YELLOW", f"Querying with: {q}")
            memories = self.dual_memory.get_relevant_memories(q, self.session.session_id, k=100)
            logger.info(f"Retrieved {len(memories)} memories for query: {q}")
            colour_print("GREEN", f"Retrieved {len(memories)} memories for query: {q}")
            all_retrieved_memories.extend(memories)
        
        logger.info(f"Total retrieved memories: {len(all_retrieved_memories)}")
        colour_print("GREEN", f"Total retrieved memories: {len(all_retrieved_memories)}")
        return all_retrieved_memories

    def _filter_by_relevance(self, memories: List[RetrievedMemory], keywords: str) -> List[RetrievedMemory]:
        relevance_filtered_mems = [
            mem for mem in memories
            if mem.relevance > 0.6
        ]
        
        # Add keyword-based memories
        with self.dual_memory.db_manager.session_scope() as session:
            keyword_messages = self.dual_memory.db_manager.get_messages_by_keywords(keywords.split(), self.session.session_id)
            for msg in keyword_messages:
                relevance_filtered_mems.append(RetrievedMemory(
                    passage=msg.content,
                    passage_info=None,
                    timestamp=msg.timestamp,
                    distance=0,
                    relevance=1.0,
                    textKeys=[],
                    metadata={"role": msg.role, "keywords": [kw.keyword for kw in msg.keywords]}
                ))
        
        logger.info(f"Memories after relevance filtering: {len(relevance_filtered_mems)}")
        colour_print("GREEN", f"Memories after relevance filtering: {len(relevance_filtered_mems)}")
        return relevance_filtered_mems

    def _calculate_relevance(self, text: str, query: str) -> float:
        # Retrieve a single memory to get its relevance score
        retrieved = self.dual_memory.semantic_memory.retrieve(query, k=1)
        if retrieved:
            return retrieved[0].relevance
        return 0.0

    def _filter_by_keywords(self, memories: List[RetrievedMemory], keywords: List[str]) -> List[RetrievedMemory]:
        keyword_filtered_mems = [
            mem for mem in memories
            if any(kw in mem.metadata.get('keywords', []) for kw in keywords)
        ]
        
        logger.info(f"Memories after keyword filtering: {len(keyword_filtered_mems)}")
        colour_print("GREEN", f"Memories after keyword filtering: {len(keyword_filtered_mems)}")
        return keyword_filtered_mems

    def _perform_spreading_activations(self, memories: List[RetrievedMemory]) -> List[RetrievedMemory]:
        logger.info(f"Performing spreading activations with {len(memories[:10])} memories.")
        colour_print("YELLOW", f"Performing spreading activations with {len(memories[:10])} memories.")
        
        secondary_memories = []
        for mem in memories[:10]:
            related_memories = self.dual_memory.semantic_memory.retrieve(mem.passage, k=5)
            for related_memory in related_memories:
                if related_memory not in secondary_memories and related_memory not in memories:
                    secondary_memories.append(related_memory)

        memories.extend(secondary_memories)
        
        logger.info(f"Memories after spreading activations: {len(memories)}")
        colour_print("GREEN", f"Memories after spreading activations: {len(memories)}")
        return memories

    def save_interaction(self, user_message: str, response_message: str, keywords: List[str]):
        user_msg = Message(role="user", content=user_message, timestamp=self.now.timestamp())
        assistant_msg = Message(role="assistant", content=response_message, timestamp=self.now.timestamp())
        
        self.session.add_interaction((user_msg, assistant_msg), keywords)

    def reset(self):
        self.dual_memory.semantic_memory.clear()
        with self.dual_memory.db_manager.session_scope() as session:
            session.query(DBMessage).delete()
            session.query(DBKeyword).delete()
        self.new_session()

    def state_as_text(self) -> str:
        state = {
            "model": self.model,
            "max_prompt_size": self.max_prompt_size,
            "max_completion_tokens": self.max_completion_tokens,
            "semantic_memory": self.dual_memory.semantic_memory.state_as_text(),
            "session": self.session.state_as_text(),
            "llm_call_idx": self.llm_call_idx
        }
        return json.dumps(state, cls=SimpleJSONEncoder)

    def from_state_text(self, state_text: str, prompt_callback: Optional[Callable[[str, str, List[Dict], str], Any]] = None):
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        self.max_prompt_size = state["max_prompt_size"]
        self.max_completion_tokens = state["max_completion_tokens"]
        self.model = state["model"]
        self.dual_memory.semantic_memory.set_state(state["semantic_memory"])
        self.session = LTMAgentSession.from_state_text(state["session"], self.dual_memory)
        self.llm_call_idx = state["llm_call_idx"]

# Helper function for logging LLM calls
def log_llm_call(run_name: str, save_name: str, debug_level: int, label: str):
    logger.debug(f"LLM call: {run_name} - {save_name} - {label}")
    # Implement additional logging logic if needed

def sanitize_filename(filename: str) -> str:
    # Remove or replace characters that are unsafe for file names
    return re.sub(r'[<>:"/\\|?*]', '-', filename)