import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    DATABASE_URL = 'sqlite:///./data/memories.db'
    LOG_FILE = './logs/app.log'
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MODEL_NAME = "meta-llama/Llama-3-70b-chat-hf"

config = Config()
