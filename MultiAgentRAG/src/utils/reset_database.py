# src/utils/reset_database.py

import sys
import os

# Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from memory.memory_manager import MemoryManager

if __name__ == "__main__":
    memory_manager = MemoryManager("memory.db", "your_api_key_here")
    memory_manager.reset_database()
    print("Database has been reset.")
