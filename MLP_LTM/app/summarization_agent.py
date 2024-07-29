from typing import List
from app.config import config
from app.utils.logging import get_logger
from app.utils.llama_tokenizer import LlamaTokenizer
from app.model_client import ModelClient

logger = get_logger('custom')

class SummarizationAgent:
    def __init__(self):
        self.model_client = ModelClient(config.MODEL_CONFIGS['summarization']['provider'])
        self.tokenizer = LlamaTokenizer()

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
        max_extractive_tokens = min(max_tokens, int(max_tokens * extractive_ratio))
        
        sorted_memories = sorted(memories, key=lambda x: self.tokenizer.count_tokens(x), reverse=True)
        
        extracted_memories = []
        current_tokens = 0
        
        for memory in sorted_memories:
            memory_tokens = self.tokenizer.count_tokens(memory)
            if current_tokens + memory_tokens <= max_extractive_tokens:
                extracted_memories.append(memory)
                current_tokens += memory_tokens
            else:
                # If we can't add any memories, at least include the first one
                if not extracted_memories:
                    truncated_memory = self.tokenizer.truncate_text(memory, max_extractive_tokens)
                    extracted_memories.append(truncated_memory)
                break
        
        return extracted_memories if extracted_memories else [self.tokenizer.truncate_text(memories[0], max_extractive_tokens)]

    async def _abstractive_summarization(self, query: str, memories: List[str], max_tokens: int) -> str:
        prompt = self._construct_summarization_prompt(query, memories)
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        
        max_new_tokens = min(
            max_tokens,
            config.MODEL_CONFIGS['summarization']['max_tokens'] - prompt_tokens - 1,
            config.MODEL_CONFIGS['summarization']['max_tokens']  # Ensure we don't exceed the model's limit
        )

        if max_new_tokens <= config.SUMMARIZATION['min_abstractive_tokens']:
            logger.warning("Not enough tokens for abstractive summarization. Returning truncated memories.")
            return self.tokenizer.truncate_text(" ".join(memories), max_tokens)

        try:
            response = self.model_client.chat_completion(
                model=config.MODEL_CONFIGS['summarization']['model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=config.MODEL_CONFIGS['summarization']['temperature']
            )
            return self.model_client.get_completion_content(response)
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