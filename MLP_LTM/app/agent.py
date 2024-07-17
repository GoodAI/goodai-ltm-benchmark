import json
import re
from together import Together
from app.db.memory_manager import MemoryManager
from app.config import config
from app.utils.logging import get_logger

logger = get_logger(__name__)

class Agent:
    def __init__(self, api_key: str, memory_manager: MemoryManager):
        self.together_client = Together(api_key=api_key)
        self.memory_manager = memory_manager

    async def process_query(self, query: str) -> str:
        # Check if the query is a special trivia request
        if self._is_trivia_request(query):
            return await self._process_trivia_request(query)
        
        # Create a new memory with just the query
        memory_id = await self.memory_manager.create_memory_with_query(query)
        
        relevant_memories = await self._retrieve_relevant_memories(query)
        response = await self._generate_response(query, relevant_memories)
        
        # Update the memory with the response
        await self.memory_manager.update_memory_with_response(memory_id, response)
        
        return response

    def _is_trivia_request(self, query: str) -> bool:
        pattern = r'^\s*Here are some trivia questions and answers for you to process\.'
        return bool(re.match(pattern, query, re.IGNORECASE))

    async def _process_trivia_request(self, query: str) -> str:
        # Extract answers from the query
        answers = re.findall(r'\bAnswer:\s*(.+?)(?=\n|$)', query, re.IGNORECASE)
        return json.dumps(answers)

    async def _retrieve_relevant_memories(self, query: str) -> list:
        relevant_memories = await self.memory_manager.get_relevant_memories(query, top_k=5)
        return relevant_memories  # Now returns formatted strings

    async def _generate_response(self, query: str, relevant_memories: list) -> str:
        prompt = self._construct_prompt(query, relevant_memories)
        response = self.together_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.MODEL_NAME,
            # max_tokens=1000  # Add this to limit the response length
        )
        return response.choices[0].message.content

#     def _construct_prompt(self, query: str, relevant_memories: list) -> str:
#         memory_context = "\n".join([f"- {memory}" for memory in relevant_memories])
#         return f"""Given the following context and query, provide a relevant and informative response:

# Context:
# {memory_context}

# Query: {query}

# Response:"""

    def _construct_prompt(self, query: str, relevant_memories: list) -> str:
        memory_context = "\n".join([f"- {memory}" for memory in relevant_memories])
        return f"""You are an advanced AI assistant with access to a database of past interactions and knowledge. Your task is to provide a relevant, informative, and logically sound response to the given query. Follow these guidelines:

    1. Context Understanding:
    - The following memories have been retrieved based on their relevance to the current query.
    - Each memory is formatted as: "<timestamp> <past_query>:<past_response>"
    - Use these memories to inform your response, but do not repeat them verbatim.

    Retrieved Memories:
    {memory_context}

    2. Query Analysis:
    - Carefully analyze the query to understand its main points, implicit assumptions, and potential complexities.
    - If the query is ambiguous or lacks crucial information, state your assumptions clearly in your response.

    3. Logical Reasoning:
    - Apply deductive, inductive, and abductive reasoning as appropriate to the query.
    - Break down complex problems into smaller, manageable parts if necessary.
    - Clearly explain your thought process and the steps you take to arrive at your conclusion.

    4. Critical Thinking:
    - Consider multiple perspectives and potential counterarguments.
    - Evaluate the reliability and relevance of the information from the retrieved memories.
    - Highlight any uncertainties or limitations in your response.

    5. Response Formulation:
    - Provide a clear, concise, and well-structured response.
    - Use examples, analogies, or hypothetical scenarios to illustrate complex concepts if appropriate.
    - If the query requires a specific format (e.g., a list, a step-by-step guide, or a numerical answer), adhere to that format.

    6. Ethical Considerations:
    - Ensure your response is ethical, unbiased, and respectful.
    - If the query touches on sensitive topics, approach them with care and objectivity.

    Now, please respond to the following query:

    Query: {query}

    Response:"""