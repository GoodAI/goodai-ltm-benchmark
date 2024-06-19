import os
import logging
from dotenv import load_dotenv
from controller import Controller
from utils.data_utils import structure_memories
from utils.json_utils import save_memory_to_json

# Ensure logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Setup logging
master_logger = logging.getLogger('master')
master_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("logs/master.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
master_logger.addHandler(file_handler)

# Console handler for query and response logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
master_logger.addHandler(console_handler)

# Separate loggers for different types of logs
chat_logger = logging.getLogger('chat')
chat_logger.setLevel(logging.DEBUG)
chat_file_handler = logging.FileHandler('logs/chat.log')
chat_file_handler.setLevel(logging.DEBUG)
chat_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
chat_logger.addHandler(chat_file_handler)

memory_logger = logging.getLogger('memory')
memory_logger.setLevel(logging.DEBUG)
memory_file_handler = logging.FileHandler('logs/memory.log')
memory_file_handler.setLevel(logging.DEBUG)
memory_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
memory_logger.addHandler(memory_file_handler)

def is_running_in_docker() -> bool:
    """Check if the code is running inside a Docker container."""
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return 'docker' in f.read()
    except Exception:
        return False

def main():
    try:
        master_logger.info("Starting the Multi-Agent RAG System")
        load_dotenv()
        openai_api_key = os.getenv("GOODAI_OPENAI_API_KEY_LTM01")
        tavily_api_key = os.getenv("TAVILY_API_KEY")

        if not openai_api_key or not tavily_api_key:
            master_logger.error("API keys not found in environment variables")
            raise EnvironmentError("API keys not found in environment variables")

        master_logger.info("API keys successfully loaded")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Determine the correct path for the memory database
        memory_db_path = "/app/memory.db" if is_running_in_docker() else "memory.db"
        master_logger.info(f"Using memory database path: {memory_db_path}")

        master_logger.debug("Initializing controller")
        controller = Controller("gpt-3.5-turbo", memory_db_path, openai_api_key)

        while True:
            try:
                query = input("Enter your query (or 'quit' to exit): ")
                if query.lower() == "quit":
                    master_logger.info("Exiting the program")
                    break

                master_logger.info(f"Executing query: {query}")
                chat_logger.info(f"Query: {query}")
                response = controller.execute_query(query)
                chat_logger.info(f"Response: {response}")
                console_handler.setLevel(logging.INFO)
                master_logger.info(f"Response: {response}")

                memories = controller.get_memories(5)
                memory_logger.info("Recent Memories:")
                for memory in memories:
                    memory_logger.info(f"Query: {memory[0]}, Result: {memory[1]}")

                structured_memories = structure_memories(memories)
                for memory in structured_memories:
                    save_memory_to_json(memory, output_dir='json_output')

            except EOFError:
                master_logger.info("Received EOF, exiting the program")
                break
            except Exception as e:
                master_logger.error(f"An error occurred while processing the query: {e}", exc_info=True)
    except Exception as e:
        master_logger.error(f"An error occurred in the main program: {e}", exc_info=True)

if __name__ == "__main__":
    main()
