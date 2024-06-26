import asyncio
from config import Config
from MultiAgentRAG.src.utils.controller import Controller
from MultiAgentRAG.src.utils.logging_setup import setup_logging
import logging

async def process_query(controller, query, chat_logger, memory_logger):
    chat_logger.info(f"Query: {query}")
    response = await controller.execute_query(query)
    chat_logger.info(f"Response: {response}")
    
    memories = await controller.get_recent_memories(5)
    memory_logger.info("Recent Memories:")
    for memory in memories:
        memory_logger.info(f"Query: {memory[0]}, Result: {memory[1]}")

async def main():
    # Initialize logging
    master_logger, chat_logger, memory_logger = setup_logging()

    # Set the log level for the memory logger
    memory_logger.setLevel(logging.DEBUG)

    try:
        master_logger.info("Starting the Multi-Agent RAG System")
        config = Config()
        
        if not config.validate_api_keys():
            raise EnvironmentError("API keys not found in environment variables")

        master_logger.info("API keys successfully loaded")
        master_logger.info(f"Using memory database path: {config.MEMORY_DB_PATH}")

        master_logger.debug("Initializing controller")
        controller = Controller(config)

        while True:
            try:
                query = input("Enter your query (or 'quit' to exit): ")
                if query.lower() == "quit":
                    master_logger.info("Exiting the program")
                    break

                master_logger.info(f"Executing query: {query}")
                await process_query(controller, query, chat_logger, memory_logger)

            except EOFError:
                master_logger.info("Received EOF, exiting the program")
                break
            except Exception as e:
                master_logger.error(f"An error occurred while processing the query: {e}", exc_info=True)
    except Exception as e:
        master_logger.error(f"An error occurred in the main program: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())