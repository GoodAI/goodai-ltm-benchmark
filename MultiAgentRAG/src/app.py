import os
import logging
import socket
from dotenv import load_dotenv
from controller import Controller
from utils.data_utils import structure_memories
from utils.json_utils import save_memory_to_json

def setup_logging(container_id: str):
    log_directory = f'logs/docker_agents/{container_id}'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    master_logger = logging.getLogger('master')
    master_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(log_directory, "master.log"))
    file_handler.setFormatter(log_formatter)
    master_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    master_logger.addHandler(console_handler)

    chat_logger = logging.getLogger('chat')
    chat_logger.setLevel(logging.DEBUG)
    chat_file_handler = logging.FileHandler(os.path.join(log_directory, 'chat.log'))
    chat_file_handler.setFormatter(log_formatter)
    chat_logger.addHandler(chat_file_handler)

    memory_logger = logging.getLogger('memory')
    memory_logger.setLevel(logging.DEBUG)
    memory_file_handler = logging.FileHandler(os.path.join(log_directory, 'memory.log'))
    memory_file_handler.setFormatter(log_formatter)
    memory_logger.addHandler(memory_file_handler)

    master_logger.info(f"Logging setup complete. Log directory: {log_directory}")
    return master_logger, chat_logger, memory_logger

def is_running_in_docker() -> bool:
    """Check if the code is running inside a Docker container."""
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return 'docker' in f.read()
    except Exception:
        return False

def main():
    container_id = socket.gethostname() if is_running_in_docker() else 'local'
    master_logger, chat_logger, memory_logger = setup_logging(container_id)

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
