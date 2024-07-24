import math
from typing import List
from together import Together
from app.config import config
from app.utils.logging import get_logger
from app.utils.llama_tokenizer import LlamaTokenizer
from app.token_manager import TokenManager

logger = get_logger('custom')

class SummarizationAgent:
    def __init__(self, api_key: str):
        self.together_client = Together(api_key=api_key)
        self.tokenizer = LlamaTokenizer()
        self.model_token_limit = config.MODEL["max_tokens"]
        self.token_manager = TokenManager(self.model_token_limit)
        self.max_input_tokens = config.MODEL["max_input_tokens"]

    async def summarize_memories(self, query: str, memories: List[str], max_tokens: int) -> List[str]:
        try:
            total_tokens = self.tokenizer.count_tokens(" ".join(memories))
            logger.debug(f"Initial memories tokens: {total_tokens}, Max tokens: {max_tokens}")
            
            if total_tokens <= max_tokens:
                logger.debug("No summarization needed")
                return memories

            extractive_summary = self._extractive_summarization(memories, max_tokens)
            extractive_tokens = self.tokenizer.count_tokens(" ".join(extractive_summary))
            logger.debug(f"After extractive summarization - Tokens: {extractive_tokens}")
            
            if extractive_tokens > max_tokens:
                logger.debug("Performing abstractive summarization")
                abstractive_summary = await self._abstractive_summarization(query, extractive_summary, max_tokens)
                abstractive_tokens = self.tokenizer.count_tokens(abstractive_summary)
                logger.debug(f"After abstractive summarization - Tokens: {abstractive_tokens}")
                
                if abstractive_tokens > max_tokens:
                    logger.warning("Abstractive summary still too long. Truncating...")
                    return [self.tokenizer.truncate_text(abstractive_summary, max_tokens)]
                return [abstractive_summary]
            
            return extractive_summary
        except Exception as e:
            logger.error(f"Error summarizing memories: {str(e)}", exc_info=True)
            raise

    def _extractive_summarization(self, memories: List[str], max_tokens: int) -> List[str]:
        extractive_ratio = config.SUMMARIZATION['extractive_ratio']
        max_extractive_tokens = min(max_tokens, config.SUMMARIZATION['max_extractive_tokens'])
        
        sorted_memories = sorted(memories, key=lambda x: self.tokenizer.count_tokens(x), reverse=True)
        
        extracted_memories = []
        current_tokens = 0
        
        for memory in sorted_memories:
            memory_tokens = self.tokenizer.count_tokens(memory)
            if current_tokens + memory_tokens <= max_extractive_tokens:
                extracted_memories.append(memory)
                current_tokens += memory_tokens
            else:
                break
        
        return extracted_memories

    async def _abstractive_summarization(self, query: str, memories: List[str], max_tokens: int) -> str:
        prompt = self._construct_summarization_prompt(query, memories)
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        
        max_new_tokens = min(max_tokens, self.model_token_limit - prompt_tokens - 1)

        if max_new_tokens <= config.SUMMARIZATION['min_abstractive_tokens']:
            logger.warning("Not enough tokens for abstractive summarization. Returning truncated memories.")
            return self.tokenizer.truncate_text(" ".join(memories), max_tokens)

        try:
            response = self.together_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=config.MODEL["model"],
                max_tokens=max_new_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating abstractive summary: {str(e)}", exc_info=True)
            raise

    def _construct_summarization_prompt(self, query: str, memories: List[str]) -> str:
        memories_text = "\n".join(memories)
        return f"""Summarize the following memories, focusing on information relevant to the given query. 
        Omit any memories that are not relevant to the query. 
        Provide an extremely concise summary that captures only the most essential points related to the query.
        Be as brief as possible while maintaining crucial information.

        Query: {query}

        Memories:
        {memories_text}

        Summary:"""