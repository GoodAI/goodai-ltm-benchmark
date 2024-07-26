from typing import List, Tuple, Dict
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
from app.db.memory_manager import MemoryManager, Memory
from app.config import config
from app.utils.logging import get_logger
from app.summarization_agent import SummarizationAgent
from app.utils.llama_tokenizer import LlamaTokenizer
from app.token_manager import TokenManager
from app.filter_agent import FilterAgent

logger = get_logger('custom')
chat_logger = get_logger('chat')

class Agent:
    def __init__(self, api_key: str, memory_manager: MemoryManager):
        self.together_client = Together(api_key=api_key)
        self.memory_manager = memory_manager
        self.summarization_agent = SummarizationAgent(api_key)
        self.filter_agent = FilterAgent(api_key)
        self.tokenizer = LlamaTokenizer()
        self.token_manager = TokenManager(config.MODEL['max_tokens'])
        self.model_token_limit = config.MODEL['max_tokens']
        self.max_input_tokens = config.MODEL['max_input_tokens']
        self.safety_margin = 1000  # Add a safety margin

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_response(self, query: str, relevant_memories: List[Memory]) -> str:
        try:
            prompt = self._construct_prompt(query, relevant_memories)
            prompt_tokens = self.tokenizer.count_tokens(prompt)
            
            max_new_tokens = self.model_token_limit - prompt_tokens - self.safety_margin - 1

            logger.debug(f"Prompt tokens: {prompt_tokens}, Max new tokens: {max_new_tokens}")
            logger.debug(f"Total tokens: {prompt_tokens + max_new_tokens}")

            if max_new_tokens <= 0:
                logger.error(f"Not enough tokens for response. Prompt tokens: {prompt_tokens}")
                raise ValueError("Input too long for model to generate a response")

            chat_logger.info(f"Generated prompt: {prompt}")
            chat_logger.info(f"Prompt tokens: {prompt_tokens}, Max new tokens: {max_new_tokens}")
            
            response = self.together_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=config.MODEL["model"],
                max_tokens=max_new_tokens
            )
            chat_logger.info(f"API response: {response}")
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    async def process_query(self, query: str) -> str:
        try:
            memory_id = await self.memory_manager.create_memory_with_query(query)
            
            relevant_memories, memory_objects, retrieval_info = await self._retrieve_relevant_memories(query, memory_id)
            
            filtered_memories = await self.filter_agent.filter_process(query, memory_objects)
            
            if not query.startswith("Here are some trivia questions and answers for you to process."):
                comparison_data = self._format_comparison_data(query, filtered_memories)
                self._write_comparison_data(comparison_data)
            
            response = await self._generate_response(query, filtered_memories)
            
            await self.memory_manager.update_memory_with_response(memory_id, response)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"An error occurred while processing your query: {str(e)}"

    async def _retrieve_relevant_memories(self, query: str, memory_id: int) -> Tuple[List[str], List[Memory], Dict[int, str]]:
        try:
            return await self.memory_manager.get_relevant_memories(query, memory_id, top_k=config.RETRIEVAL['top_k'])
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {str(e)}", exc_info=True)
            raise

    def _construct_prompt(self, query: str, relevant_memories: List[Memory]) -> str:
        memory_context = ""
        if relevant_memories:
            memory_count = len(relevant_memories)
            memory_count_word = self._num2words(memory_count)
            memory_context = f"""
        Retrieved Memories:
        You have been provided {memory_count_word} relevant memor{'y' if memory_count == 1 else 'ies'}:
        {chr(10).join([f"- {self._format_memory_for_prompt(memory)}" for memory in relevant_memories])}
        """

        return f"""You are an AI assistant designed to provide accurate and relevant responses. Use the following guidelines:

        1. Context:
        - Relevant past interactions are provided below, if any.
        - Format: "<timestamp> <past_query>:<past_response>"
        - Use this context to inform your response without directly repeating it.
        - If there are no memories provided, there is nothing relevant in your memory banks. Keep this to yourself.
        - In general, the user should not be made aware of your memory capabilities.
    {memory_context}
        2. Response Guidelines:
        - Analyze the query thoroughly, addressing its main points and any implicit aspects.
        - If the query lacks crucial information, state your assumptions clearly.
        - Apply logical reasoning to formulate your response.
        - Consider multiple perspectives and evaluate the reliability of the provided context.
        - Provide a clear, concise, and well-structured response.
        - If a specific format is required (e.g., list, steps, numerical answer), adhere to it.
        - You are terse and pithy, not as a personality trait but to be more economical with your token usage, but do not let this impact your specificity. 
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

    def _format_memory_for_prompt(self, memory: Memory) -> str:
        return f"{memory.timestamp} {memory.query}:{memory.response}" if memory.response else f"{memory.timestamp} {memory.query}:"

    def _format_comparison_data(self, query: str, memories: List[Memory]) -> Dict:
        return {
            "query": query,
            "memories": [self._format_memory(m) for m in memories]
        }

    def _format_memory(self, memory: Memory) -> Dict:
        return {
            "id": memory.id,
            "query": memory.query,
            "response": memory.response,
            "timestamp": memory.timestamp
        }

    def _write_comparison_data(self, data: Dict):
        os.makedirs("comparison_data", exist_ok=True)
        file_name = "comparison_data/comparison_data.json"
        
        try:
            # If file exists, read existing data
            if os.path.exists(file_name):
                with open(file_name, "r") as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Add timestamp to the new data
            data['timestamp'] = int(time.time())

            # Append new data
            existing_data.append(data)

            # Write updated data back to file
            with open(file_name, "w") as f:
                json.dump(existing_data, f, indent=2)
            
            logger.info(f"Comparison data appended to {file_name}")
        except Exception as e:
            logger.error(f"Error writing comparison data: {str(e)}", exc_info=True)

    @staticmethod
    def _num2words(num: int) -> str:
        """Convert small numbers to words."""
        units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num < 10:
            return units[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            return tens[num // 10] + ("-" + units[num % 10] if num % 10 != 0 else "")
        else:
            return str(num)  # For numbers 100 and above, return as string