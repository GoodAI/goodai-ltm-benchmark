from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field


load_dotenv()

class Config(BaseSettings):
    TOGETHER_API_KEY: str = Field(..., env="TOGETHER_API_KEY")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")  # Add this line
    DATABASE_URL: str = 'sqlite:///./data/memories.db'
    LOG_FILE_MAIN: str = './logs/app.log'
    LOG_FILE_CUSTOM: str = './logs/custom.log'
    LOG_FILE_CHAT: str = './logs/chat.log'
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    MODEL_NAME: str = "meta-llama/Llama-3-70b-chat-hf"

    MEMORY_LINKING: dict = {
        'enabled': True,
        'similarity_threshold': 0.8,
        'max_links_per_memory': None,  # None means infinite
        'query_only_linking': True,
        'keyword_matching': {
            'enabled': False,
            'threshold': 0.7
        }
    }

    EMBEDDING: dict = {
        'model': "text-embedding-ada-002",
        'dimensions': 1536  # Dimensions for the chosen embedding model
    }

    RETRIEVAL: dict = {
        'top_k': None,  # None means retrieve all relevant memories
        'min_similarity': 0.65
    }

    MEMORY_FORMATTING: dict = {
        'timestamp_format': '%Y-%m-%d %H:%M:%S'
    }

    class Config:
        env_file = ".env"

config = Config()
