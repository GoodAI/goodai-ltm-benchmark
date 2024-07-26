from typing import List
from app.db.memory_database import Memory
from app.config import config
from app.utils.logging import get_logger
from together import Together

logger = get_logger('custom')

class FilterAgent:
    def __init__(self, api_key: str):
        self.together_client = Together(api_key=api_key)
        self.model = config.FILTER_MODEL['model']

    async def forgetting_check(self, query: str) -> bool:
        try:
            prompt = f"""Analyze the following query and determine if it contains an explicit request to forget or disregard specific information:

            Query: {query}

            Respond with 'Yes' if the query contains a forget/disregard request, or 'No' if it doesn't. It is imperative that your response is only 'Yes' or 'No'."""

            response = self.together_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=10,
                temperature=config.FILTER_MODEL['temperature']
            )

            result = response.choices[0].message.content.strip().lower()
            return result == 'yes'
        except Exception as e:
            logger.error(f"Error in forgetting_check: {str(e)}", exc_info=True)
            raise

    async def filter_memory(self, query: str, memory: Memory) -> bool:
        try:
            prompt = f"""Analyze the following pair of queries, then determine if the Query_one is remotely relevant to the Query_two:

            Relevancy is defined as does Query_two provide ANY value to Query_one

            Query_one: '{query}'
            Query_two: '{memory.query}'

            Respond with 'Yes' if the queries are relevant to each other, or 'No' if it isn't. It is imperative that your response is only 'Yes' or 'No'."""

            response = self.together_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=10,
                temperature=config.FILTER_MODEL['temperature']
            )

            result = response.choices[0].message.content.strip().lower()
            return result == 'yes'
        except Exception as e:
            logger.error(f"Error in filter_memory: {str(e)}", exc_info=True)
            raise

    async def filter_process(self, query: str, retrieved_memories: List[Memory]) -> List[Memory]:
        try:
            filtered_memories = []
            forget_pruned_filtered_memories = []
            
            for memory in retrieved_memories:
                is_relevant = await self.filter_memory(query, memory)
                if is_relevant:
                    filtered_memories.append(memory)
            
            for memory in filtered_memories:        
                if await self.forgetting_check(memory.query):
                    logger.info(f"Forgetting check triggered for memory ID {memory.id}")
                    break
                forget_pruned_filtered_memories.append(memory)
            
            logger.info(f"Filtered {len(retrieved_memories)} memories down to {len(forget_pruned_filtered_memories)}")
            return forget_pruned_filtered_memories
        except Exception as e:
            logger.error(f"Error in filter_process: {str(e)}", exc_info=True)
            raise