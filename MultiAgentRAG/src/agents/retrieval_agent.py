# # src/agents/retrieval_agent.py

# import logging
# from typing import List, Tuple

# logger = logging.getLogger('master')

# class RetrievalAgent:
#     def __init__(self, memory_manager):
#         self.memory_manager = memory_manager

#     def retrieve(self, query: str) -> List[Tuple[str, str, str]]:
#         relevant_memories = self.memory_manager.retrieve_relevant_memories(query)
#         logger.debug(f"Retrieved and ranked relevant memories for query: {query}")
#         return relevant_memories

