from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

load_dotenv()

class Config(BaseSettings):
    TOGETHER_API_KEY: str = Field(..., env="TOGETHER_API_KEY")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    DATABASE_URL: str = 'sqlite:///./data/memories.db'
    LOG_FILE_MAIN: str = './logs/app.log'
    LOG_FILE_CUSTOM: str = './logs/custom.log'
    LOG_FILE_CHAT: str = './logs/chat.log'
    LOG_LEVEL: str = Field(default="DEBUG", env="LOG_LEVEL")

    MODEL: dict = {
        'model': "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        'max_tokens': 32768,
        'reserved_tokens': 1000,
        'max_input_tokens': 31768  # Maximum tokens for input (max_tokens - reserved_tokens - 1)
    }

    MEMORY_LINKING: dict = {
        'enabled': True,
        'similarity_threshold': 0.6,
        'max_links_per_memory': None,
        'query_only_linking': True,
        'keyword_matching': {
            'enabled': True,
            'embedding_weight': 0.7,
            'keyword_weight': 0.3,
        }
    }

    EMBEDDING: dict = {
        'model': "text-embedding-ada-002",
        'dimensions': 1536
    }

    RETRIEVAL: dict = {
        'top_k': None,
        'min_similarity': 0.8
    }

    MEMORY_FORMATTING: dict = {
        'timestamp_format': '%Y-%m-%d %H:%M:%S'
    }

    SUMMARIZATION: dict = {
        'extractive_ratio': 0.5,
        'max_extractive_tokens': 1000,
        'min_abstractive_tokens': 100
    }

    class Config:
        env_file = ".env"

config = Config()