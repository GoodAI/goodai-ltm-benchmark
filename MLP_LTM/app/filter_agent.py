from typing import List
import asyncio
from app.db.memory_database import Memory
from app.config import config
from app.utils.logging import get_logger
from app.model_client import ModelClient

logger = get_logger('custom')

class FilterAgent:
    def __init__(self):
        self.model_client = ModelClient(config.MODEL_CONFIGS['filter']['provider'])
        self.relevance_threshold = 1  # Lowered to include more remotely relevant memories

    async def calculate_relevance_score(self, query: str, memory: Memory) -> float:
        try:
            prompt = f"""TASK: Evaluate the relevance of QUERY_2 to QUERY_1.

[QUERY_1]: {query}
[QUERY_2]: {memory.query}

RELEVANCE DEFINITION: QUERY_2 provides ANY value in addressing QUERY_1.

SCORING GUIDE:
0 = No relevance
1 = Very low relevance
2 = Low relevance
3 = Moderate relevance
4 = High relevance
5 = Very high relevance

REASONING STEPS:
1. Identify the main topic/intent of QUERY_1.
2. Identify the main topic/intent of QUERY_2.
3. Compare the topics/intents for any overlaps or connections.
4. Assess how information from QUERY_2 could contribute to QUERY_1.
5. Determine the relevance score based on this assessment.

EXAMPLES:
[QUERY_1]: "What's the weather like today?"
[QUERY_2]: "What's the capital of France?"
REASONING:
1. QUERY_1 is about current weather conditions.
2. QUERY_2 is about geography/capital cities.
3. No apparent overlap in topics.
4. Information about France's capital doesn't contribute to weather information.
5. Score: 0 (No relevance)

[QUERY_1]: "How do I bake a chocolate cake?"
[QUERY_2]: "What's the best type of chocolate for baking?"
REASONING:
1. QUERY_1 is about baking a chocolate cake.
2. QUERY_2 is about chocolate selection for baking.
3. Both queries involve baking and chocolate.
4. Information about the best baking chocolate directly contributes to making a chocolate cake.
5. Score: 4 (High relevance)

[QUERY_1]: "What are the symptoms of the flu?"
[QUERY_2]: "How does the immune system work?"
REASONING:
1. QUERY_1 is about flu symptoms.
2. QUERY_2 is about the immune system's function.
3. There's a connection as the immune system responds to flu.
4. Understanding the immune system provides context for flu symptoms, but doesn't directly list them.
5. Score: 2 (Low relevance)

OUTPUT INSTRUCTION: Provide ONLY the numerical score (0-5) as the final answer. Do not include any explanations or additional text."""

            response = await self.model_client.chat_completion(
                model=config.MODEL_CONFIGS['filter']['model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.MODEL_CONFIGS['filter']['max_tokens'],
                temperature=0.0
            )

            result = self.model_client.get_completion_content(response)
            score = float(result.strip())
            
            # Ensure the score is within the valid range
            return max(0, min(10, score))

        except ValueError:
            logger.error("Failed to parse relevance score from the model")
            return 0  # Return lowest score in case of parsing errors
        except Exception as e:
            logger.error(f"Error in calculate_relevance_score: {str(e)}", exc_info=True)
            return 0  # Return lowest score in case of any errors

    async def filter_process(self, query: str, retrieved_memories: List[Memory]) -> List[Memory]:
        try:
            tasks = [self.calculate_relevance_score(query, memory) for memory in retrieved_memories]
            relevance_scores = await asyncio.gather(*tasks)

            filtered_memories = []
            for memory, score in zip(retrieved_memories, relevance_scores):
                logger.debug(f"Relevance score for memory {memory.id}: {score}")
                if score >= self.relevance_threshold:
                    filtered_memories.append(memory)
            
            logger.info(f"Filtered {len(retrieved_memories)} memories down to {len(filtered_memories)}")
            return filtered_memories
        except Exception as e:
            logger.error(f"Error in filter_process: {str(e)}", exc_info=True)
            return retrieved_memories  # Return all memories in case of an error