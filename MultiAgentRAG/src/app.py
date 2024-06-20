import os
from dotenv import load_dotenv
from controller import Controller
from utils.data_utils import structure_memories
from utils.json_utils import save_memory_to_json
from logging_setup import setup_logging, is_running_in_docker  # Import logging setup

def main():
    master_logger, chat_logger, memory_logger = setup_logging()

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
