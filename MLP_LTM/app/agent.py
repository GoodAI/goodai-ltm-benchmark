from typing import List, Tuple, Dict
from together import Together
from app.db.memory_manager import MemoryManager, Memory
from app.config import config
from app.utils.logging import get_logger
from num2words import num2words
from app.summarization_agent import SummarizationAgent
from app.utils.llama_tokenizer import LlamaTokenizer
from app.token_manager import TokenManager

logger = get_logger('custom')
chat_logger = get_logger('chat')

class Agent:
    def __init__(self, api_key: str, memory_manager: MemoryManager):
        self.together_client = Together(api_key=api_key)
        self.memory_manager = memory_manager
        self.summarization_agent = SummarizationAgent(api_key)
        self.tokenizer = LlamaTokenizer()
        self.token_manager = TokenManager(config.MODEL['model'])
        self.model_token_limit = config.MODEL['max_tokens']
        self.max_input_tokens = config.MODEL['max_input_tokens']
        self.safety_margin = 1000  # Add a safety margin

    async def process_query(self, query: str) -> str:
        try:
            memory_id = await self.memory_manager.create_memory_with_query(query)
            
            relevant_memories, memory_objects, retrieval_info = await self._retrieve_relevant_memories(query, memory_id)
            response = await self._generate_response(query, relevant_memories)
            
            await self.memory_manager.update_memory_with_response(memory_id, response)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    async def _retrieve_relevant_memories(self, query: str, memory_id: int) -> Tuple[List[str], List[Memory], Dict[int, str]]:
        try:
            query_embedding = await self.memory_manager.embedding_service.get_embedding(query)
            formatted_memories, memory_objects, retrieval_info = await self.memory_manager.get_relevant_memories(query, memory_id, top_k=config.RETRIEVAL['top_k'])
            
            if config.MEMORY_LINKING['query_only_linking']:
                self.memory_manager.memory_linker.update_links_for_query(query_embedding, memory_objects)
            
            empty_prompt = self._construct_prompt(query, [])
            prompt_overhead = self.tokenizer.count_tokens(empty_prompt)
            available_tokens = self.max_input_tokens - prompt_overhead - config.MODEL['reserved_tokens'] - self.safety_margin

            logger.debug(f"Prompt overhead tokens: {prompt_overhead}")
            logger.debug(f"Available tokens for memories: {available_tokens}")

            memories_tokens = self.tokenizer.count_tokens(" ".join(formatted_memories))
            logger.debug(f"Initial memories tokens: {memories_tokens}")

            while memories_tokens > available_tokens:
                logger.debug("Summarization needed")
                formatted_memories = await self.summarization_agent.summarize_memories(query, formatted_memories, available_tokens)
                memories_tokens = self.tokenizer.count_tokens(" ".join(formatted_memories))
                logger.debug(f"After summarization - Memories tokens: {memories_tokens}")
                available_tokens = int(available_tokens * 0.9)  # Reduce available tokens by 10% each iteration

            logger.info(f"Final memory token count: {memories_tokens}")
            return formatted_memories, memory_objects, retrieval_info
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {str(e)}", exc_info=True)
            raise

    async def _generate_response(self, query: str, relevant_memories: List[str]) -> str:
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

    def _construct_prompt(self, query: str, relevant_memories: List[str]) -> str:
        memory_context = ""
        if relevant_memories:
            memory_count = len(relevant_memories)
            memory_count_word = num2words(memory_count)
            memory_context = f"""
        Retrieved Memories:
        You have been provided {memory_count_word} relevant memor{'y' if memory_count == 1 else 'ies'}:
        {chr(10).join([f"- {memory}" for memory in relevant_memories])}
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