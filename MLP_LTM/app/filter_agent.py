from typing import List
from app.db.memory_database import Memory
from app.config import config
from app.utils.logging import get_logger
from app.model_client import ModelClient

logger = get_logger('custom')

class FilterAgent:
    def __init__(self):
        self.model_client = ModelClient(config.MODEL_CONFIGS['filter']['provider'])

    # async def forgetting_check(self, query: str) -> bool:
    #     try:
    #         prompt = f"""Analyze the following query and determine if it contains an explicit request to forget or disregard specific information:

    #         Query: {query}

    #         Respond with 'Yes' if the query contains a forget/disregard request, or 'No' if it doesn't. It is imperative that your response is only 'Yes' or 'No'."""

    #         response = self.model_client.chat_completion(
    #             model=config.MODEL_CONFIGS['filter']['model'],
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=config.MODEL_CONFIGS['filter']['max_tokens'],
    #             temperature=config.MODEL_CONFIGS['filter']['temperature']
    #         )

    #         result = self.model_client.get_completion_content(response).strip().lower()
    #         return result == 'yes'
    #     except Exception as e:
    #         logger.error(f"Error in forgetting_check: {str(e)}", exc_info=True)
    #         raise

    async def filter_memory(self, query: str, memory: str) -> bool:
        try:
            prompt = f"""Analyze the following query and memory to determine if the memory is relevant and should be retained:

Query: {query}

Memory: {memory.query}

Instructions:
1. Evaluate the semantic relationship between the query and the memory.
2. Consider the following criteria:
a. Topical relevance: Does the memory relate to the same subject matter as the query?
b. Contextual importance: Does the memory provide context that could be useful in understanding or addressing the query?
c. Temporal relevance: Is the memory recent or timeless enough to be applicable to the query?
d. Factual connection: Does the memory contain facts, data, or information that directly or indirectly supports the query?
3. Ignore superficial keyword matches that don't capture true relevance.
4. Do not discard memories simply because they contain additional information beyond the query's scope.
5. Err on the side of retention if there's any doubt about relevance.

Respond with 'Retain' if the memory should be kept, or 'Discard' if it's irrelevant. Your response must be only one word: either 'Retain' or 'Discard'."""

            response = self.model_client.chat_completion(
                model=config.MODEL_CONFIGS['filter']['model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.MODEL_CONFIGS['filter']['max_tokens'],
                temperature=0.0  # Set to 0 for deterministic output
            )

            result = self.model_client.get_completion_content(response).strip().lower()
            return result == 'retain'
        except Exception as e:
            logger.error(f"Error in memory_relevance_check: {str(e)}", exc_info=True)
            raise

    # async def filter_memory(self, query: str, memory: Memory) -> bool:
    #     try:
    #         prompt = f"""Analyze the following pair of queries, then determine if the Query_one is remotely relevant to the Query_two:

    #         Relevancy is defined as does Query_two provide ANY value to Query_one

    #         Query_one: '{query}'
    #         Query_two: '{memory.query}'

    #         Respond with 'Yes' if the queries are relevant to each other, or 'No' if it isn't. It is imperative that your response is only 'Yes' or 'No'."""

    #         response = self.model_client.chat_completion(
    #             model=config.MODEL_CONFIGS['filter']['model'],
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=config.MODEL_CONFIGS['filter']['max_tokens'],
    #             temperature=config.MODEL_CONFIGS['filter']['temperature']
    #         )

    #         result = self.model_client.get_completion_content(response).strip().lower()
    #         return result == 'yes'
    #     except Exception as e:
    #         logger.error(f"Error in filter_memory: {str(e)}", exc_info=True)
    #         raise

    async def filter_process(self, query: str, retrieved_memories: List[Memory]) -> List[Memory]:
        try:
            filtered_memories = []
            # forget_pruned_filtered_memories = []
            
            for memory in retrieved_memories:
                is_relevant = await self.filter_memory(query, memory)
                if is_relevant:
                    filtered_memories.append(memory)
            
            # for memory in filtered_memories:        
            #     if await self.forgetting_check(memory.query):
            #         logger.info(f"Forgetting check triggered for memory ID {memory.id}")
            #         break
            #     forget_pruned_filtered_memories.append(memory)
            
            # logger.info(f"Filtered {len(retrieved_memories)} memories down to {len(forget_pruned_filtered_memories)}")
            logger.info(f"Filtered {len(retrieved_memories)} memories down to {len(filtered_memories)}")

            # return forget_pruned_filtered_memories
            return filtered_memories
        except Exception as e:
            logger.error(f"Error in filter_process: {str(e)}", exc_info=True)
            raise