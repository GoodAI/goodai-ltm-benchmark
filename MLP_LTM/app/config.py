import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    DATABASE_URL = 'sqlite:///./data/memories.db'
    LOG_FILE = './logs/app.log'
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MODEL_NAME = "meta-llama/Llama-3-70b-chat-hf"

    # Loadout system
    MEMORY_LINKING = {
        'enabled': True,
        'similarity_threshold': 0.8,
        'max_links_per_memory': None,  # None means infinite
        'query_only_linking': False,
        'keyword_matching': {
            'enabled': False,
            'threshold': 0.7
        }
    }

    EMBEDDING = {
        'model': "text-embedding-ada-002",
        'dimensions': 1536  # Dimensions for the chosen embedding model
    }

    RETRIEVAL = {
        'top_k': None,  # None means retrieve all relevant memories
        'min_similarity': 0.8
    }

    MEMORY_FORMATTING = {
        'timestamp_format': '%Y-%m-%d %H:%M:%S'
    }

config = Config()