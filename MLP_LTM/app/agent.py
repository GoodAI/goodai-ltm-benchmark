# app/agent.py
import json
import re
from typing import List
from together import Together
from app.db.memory_manager import MemoryManager
from app.config import config
from app.utils.logging import get_logger

logger = get_logger('custom')
chat_logger = get_logger('chat')

class Agent:
    def __init__(self, api_key: str, memory_manager: MemoryManager):
        self.together_client = Together(api_key=api_key)
        self.memory_manager = memory_manager

    async def process_query(self, query: str) -> str:
        try:
            if self._is_trivia_request(query):
                return await self._process_trivia_request(query)
            
            memory_id = await self.memory_manager.create_memory_with_query(query)
            
            relevant_memories = await self._retrieve_relevant_memories(query, memory_id)
            response = await self._generate_response(query, relevant_memories)
            
            await self.memory_manager.update_memory_with_response(memory_id, response)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    def _is_trivia_request(self, query: str) -> bool:
        pattern = r'^\s*Here are some trivia questions and answers for you to process\.'
        return bool(re.match(pattern, query, re.IGNORECASE))

    async def _process_trivia_request(self, query: str) -> str:
        try:
            answers = re.findall(r'\bAnswer:\s*(.+?)(?=\n|$)', query, re.IGNORECASE)
            return json.dumps(answers)
        except Exception as e:
            logger.error(f"Error processing trivia request: {str(e)}", exc_info=True)
            raise

    async def _retrieve_relevant_memories(self, query: str, memory_id: int) -> List[str]:
        try:
            relevant_memories = await self.memory_manager.get_relevant_memories(query, memory_id, top_k=config.RETRIEVAL['top_k'])
            return relevant_memories
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {str(e)}", exc_info=True)
            raise

    async def _generate_response(self, query: str, relevant_memories: List[str]) -> str:
        try:
            prompt = self._construct_prompt(query, relevant_memories)
            chat_logger.info(f"Generated prompt: {prompt}")
            
            response = self.together_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=config.MODEL_NAME,
            )
            chat_logger.info(f"API response: {response}")
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    def _construct_prompt(self, query: str, relevant_memories: List[str]) -> str:
        memory_context = "\n".join([f"- {memory}" for memory in relevant_memories])
        return f"""You are an AI assistant designed to provide accurate and relevant responses. Use the following guidelines:

    1. Context:
    - Relevant past interactions are provided below.
    - Format: "<timestamp> <past_query>:<past_response>"
    - Use this context to inform your response without directly repeating it.
    - If there are no context provided there is nothing relevant in your memory banks, keep this to yourself.
    - in general, the user should not be made aware of your memory capabilities. 

    Retrieved Memories:
    {memory_context}

    2. Response Guidelines:
    - Analyze the query thoroughly, addressing its main points and any implicit aspects.
    - If the query lacks crucial information, state your assumptions clearly.
    - Apply logical reasoning to formulate your response.
    - Consider multiple perspectives and evaluate the reliability of the provided context.
    - Provide a clear, concise, and well-structured response.
    - If a specific format is required (e.g., list, steps, numerical answer), adhere to it.
    - You are terse and pithy, not as a personality trait but to be more economical with you token usage.
    - Avoid unnecessary affirmations or filler phrases at the beginning of your response.

    3. Memory Management:
    - If asked to remember or forget specific information, acknowledge this request in your response.
    - If asked about previous interactions, use the provided context to inform your answer.

    4. Task Handling:
    - For multi-step tasks, offer to complete them incrementally and seek feedback.
    - If unable to perform a task, state this directly without apologizing.

    5. Special Cases:
    - For questions about events after April 2024, respond as a well-informed individual from April 2024 would.
    - If asked about controversial topics, provide careful thoughts and clear information without explicitly labeling the topic as sensitive.

    Now, please respond to the following query:

    Query: {query}

    Response:"""