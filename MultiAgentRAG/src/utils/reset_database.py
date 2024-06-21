# src/utils/reset_database.py

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the base directory of the project to the PYTHONPATH
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(base_path)

from src.memory.memory_manager import MemoryManager
from config import Config  # Import the config

if __name__ == "__main__":
    memory_manager = MemoryManager(Config.MEMORY_DB_PATH, Config.OPENAI_API_KEY)
    memory_manager.reset_database()
    print("Database has been reset.")
