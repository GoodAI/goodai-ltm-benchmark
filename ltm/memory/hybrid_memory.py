from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.base import RetrievedMemory
from sqlalchemy.orm import object_mapper
from ..database.manager import DatabaseManager
from typing import List, Tuple, Optional, Union
from model_interfaces.base_ltm_agent import Message
import json

class HybridMemory:
    def __init__(self, db_url: str, semantic_memory_config: TextMemoryConfig, max_retrieve_capacity: int = 2000):
        self.db_manager = DatabaseManager(db_url)
        self.semantic_memory = AutoTextMemory.create(config=semantic_memory_config)
        self.max_retrieve_capacity = max_retrieve_capacity
        
    def create_session(self, session_id: str, created_at: float):
        return self.db_manager.create_session(session_id, created_at)

    def add_interaction(self, session_id: str, user_message: str, assistant_message: str, timestamp: float, keywords: list[str]):
        user_semantic_key = self.semantic_memory.add_text(user_message, timestamp=timestamp, metadata={"keywords": keywords})
        self.semantic_memory.add_separator()
        assistant_semantic_key = self.semantic_memory.add_text(assistant_message, timestamp=timestamp, metadata={"keywords": keywords})
        self.semantic_memory.add_separator()

        self.db_manager.add_message(session_id, "user", user_message, timestamp, str(user_semantic_key))
        self.db_manager.add_message(session_id, "assistant", assistant_message, timestamp, str(assistant_semantic_key))

    def get_relevant_memories(self, query: str, all_keywords: List[str], cost_cb, k: int = 100) -> List[Tuple[Message, Message]]:
        all_retrieved_memories = []
        for q in [query]:  # You might want to add more queries here
            all_retrieved_memories.extend(self.semantic_memory.retrieve(q, k=k))
        
        # Relevance and keyword filtering
        relevance_filtered_mems = [x for x in all_retrieved_memories if x.relevance > 0.6]
        keyword_filtered_mems = self.retrieve_from_keywords(all_keywords)
        
        combined_memories = relevance_filtered_mems + keyword_filtered_mems
        
        # Additional keyword filtering
        keyword_filtered_mems = []
        for m in combined_memories:
            if any(kw in m.metadata.get("keywords", []) for kw in all_keywords):
                keyword_filtered_mems.append(m)
        
        # Spreading activation
        for mem in keyword_filtered_mems[:10]:
            for r_mem in self.semantic_memory.retrieve(mem.passage, k=5):
                if r_mem.relevance > 0.6 and r_mem not in keyword_filtered_mems:
                    keyword_filtered_mems.append(r_mem)
        
        # Convert RetrievedMemory objects to Message tuples
        converted_results = []
        for mem in keyword_filtered_mems:
            interaction = self.get_interaction_by_semantic_key(str(mem.textKeys[0]))
            if interaction:
                converted_results.append(interaction)
        
        # Implement LLM memory filter here (you'll need to add this method to the class)
        filtered_interactions = self.llm_memory_filter(converted_results, [query], cost_cb)
        
        return filtered_interactions

    def get_interaction_by_semantic_key(self, semantic_key: str) -> Optional[Tuple[Message, Message]]:
        try:
            interaction = self.db_manager.get_interaction_by_semantic_key(semantic_key)
            if interaction:
                user_message = Message(
                    role=interaction[0].role,
                    content=interaction[0].content,
                    timestamp=interaction[0].timestamp
                )
                assistant_message = Message(
                    role=interaction[1].role,
                    content=interaction[1].content,
                    timestamp=interaction[1].timestamp
                )
                return (user_message, assistant_message)
        except Exception as e:
            print(f"Error retrieving interaction: {e}")
        return None

    def get_interaction_by_timestamp(self, session_id: str, timestamp: float) -> Optional[Tuple[Message, Message]]:
        messages = self.db_manager.get_messages_by_timestamp(session_id, timestamp)
        if len(messages) == 2:
            def to_dict(obj):
                return {
                    'role': obj.role,
                    'content': obj.content,
                    'timestamp': obj.timestamp
                }
            
            return (Message(**to_dict(messages[0])), Message(**to_dict(messages[1])))
        return None

    def get_recent_messages(self, session_id: str, limit: int) -> List[Message]:
        messages = self.db_manager.get_recent_messages(session_id, limit)
        return [Message(role=m.role, content=m.content, timestamp=m.timestamp) for m in messages]

    def get_all_messages(self, session_id: str) -> List[Message]:
        messages = self.db_manager.get_all_messages(session_id)
        return [Message(role=m.role, content=m.content, timestamp=m.timestamp) for m in messages]

    def get_session(self, session_id: str):
        return self.db_manager.get_session(session_id)

    def is_empty(self) -> bool:
        return self.semantic_memory.is_empty()

    def clear(self):
        self.semantic_memory.clear()
        self.db_manager.clear_all()

    def state_as_text(self) -> str:
        try:
            return json.dumps({
                "semantic_memory": self.semantic_memory.state_as_text(),
                "database": self.db_manager.export_data()
            })
        except Exception as e:
            print(f"Error in state_as_text: {e}")
            return json.dumps({"error": str(e)})

    def set_state(self, state_text: str):
        try:
            state = json.loads(state_text)
            if "error" in state:
                print(f"Error in previous state: {state['error']}")
                return
            self.semantic_memory.set_state(state["semantic_memory"])
            self.db_manager.import_data(state["database"])
        except Exception as e:
            print(f"Error setting state: {e}")

    def retrieve_from_keywords(self, keywords: List[str]) -> List[RetrievedMemory]:
        all_memories = self.semantic_memory.retrieve("", k=2000)
        selected_mems = []
        for m in all_memories:
            if any(kw in m.metadata.get("keywords", []) for kw in keywords):
                selected_mems.append(m)
        return selected_mems

    def get_message_count(self, session_id: str) -> int:
        return self.db_manager.get_message_count(session_id)
    