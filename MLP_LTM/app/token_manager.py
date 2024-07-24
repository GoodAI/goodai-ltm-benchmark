# app/token_manager.py

from typing import List, Tuple
from app.utils.llama_tokenizer import LlamaTokenizer

class TokenManager:
    def __init__(self, model_token_limit: int, reserved_tokens: int = 1000):
        self.model_token_limit = model_token_limit
        self.reserved_tokens = reserved_tokens
        self.tokenizer = LlamaTokenizer()

    def allocate_tokens(self, query: str, memories: List[str]) -> Tuple[int, int]:
        query_tokens = self.tokenizer.count_tokens(query)
        available_tokens = self.model_token_limit - query_tokens - self.reserved_tokens
        memory_tokens = sum(self.tokenizer.count_tokens(memory) for memory in memories)
        
        if memory_tokens <= available_tokens:
            return available_tokens, 0
        
        summarization_tokens = available_tokens // 2
        response_tokens = available_tokens - summarization_tokens
        return summarization_tokens, response_tokens