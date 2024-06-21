# config.py

import os

class Config:
    # General settings
    MODEL_NAME = "gpt-3.5-turbo"
    
    # Database settings
    MEMORY_DB_PATH = "/app/memory.db" if os.path.exists("/.dockerenv") else "memory.db"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("GOODAI_OPENAI_API_KEY_LTM01")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Other settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
    
    # Processing agent settings
    PROCESSING_AGENT_MEMORIES_INCLUDED = 3  # Number of memories to include in context

    MEMORY_RETRIEVAL_THRESHOLD = 0.75  # Similarity threshold for memory retrieval
    MEMORY_RETRIEVAL_LIMIT = 10