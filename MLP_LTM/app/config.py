from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

load_dotenv()

class Config(BaseSettings):
    TOGETHER_API_KEY: str = Field(..., env="TOGETHER_API_KEY")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    DATABASE_URL: str = 'sqlite:///./data/memories.db'
    LOG_FILE_MAIN: str = './logs/app.log'
    LOG_FILE_CUSTOM: str = './logs/custom.log'
    LOG_FILE_CHAT: str = './logs/chat.log'
    LOG_LEVEL: str = Field(default="DEBUG", env="LOG_LEVEL")

    # MODEL_CONFIGS: dict = {
    #     'main': {
    #         'provider': 'groq',
    #         'model': 'llama-3.1-70b-versatile',
    #         'max_tokens': 8000,  # Updated to Groq's limit
    #         'temperature': 0.7,
    #     },
    #     'filter': {
    #         'provider': 'groq',
    #         'model': 'llama-3.1-8b-instant',
    #         'max_tokens': 10,
    #         'temperature': 0,
    #     },
    #     'summarization': {
    #         'provider': 'groq',
    #         'model': 'llama-3.1-70b-versatile',
    #         'max_tokens': 500,
    #         'temperature': 0.3,
    #     }
    # }
    # MODEL_CONFIGS: dict = {
    #     'main': {
    #         'provider': 'together',
    #         'model': 'meta-llama/Llama-3-70b-chat-hf',
    #         'max_tokens': 8000,  # Updated to Groq's limit
    #         'temperature': 0.05,
    #     },
    #     'filter': {
    #         'provider': 'together',
    #         'model': 'meta-llama/Llama-3-70b-chat-hf',
    #         'max_tokens': 10,
    #         'temperature': 0,
    #     },
    #     'summarization': {
    #         'provider': 'together',
    #         'model': 'meta-llama/Llama-3-70b-chat-hf',
    #         'max_tokens': 500,
    #         'temperature': 0,
    #     }
    # }

    MODEL_CONFIGS: dict = {
        'main': {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'max_tokens': 4096,
            'temperature': 0,
        },
        'filter': {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'max_tokens': 10,
            'temperature': 0,
        },
        'summarization': {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'max_tokens': 500,
            'temperature': 0,
        }
    }

    MEMORY_LINKING: dict = {
        'enabled': False,
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
        'top_k': 5,
        'min_similarity': 0.68
    }

    MEMORY_FORMATTING: dict = {
        'timestamp_format': '%Y-%m-%d %H:%M:%S'
    }

    SUMMARIZATION: dict = {
        'extractive_ratio': 0.8,
        'min_abstractive_tokens': 100
    }

    class Config:
        env_file = ".env"

config = Config()
