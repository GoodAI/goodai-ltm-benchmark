import asyncio
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging
from src.utils.structured_logging import get_logger

loggers = setup_logging()
logger = get_logger('master')

controller = Controller()
similarity_analyzer = None

async def initialize():
    await controller.initialize()
    global similarity_analyzer

async def main():
    await initialize()
    while True:
        try:
            command = input("Enter a command (query/stats/distribution/visualize/quit): ").strip().lower()
            logger.info("Received command", extra={"command": command})
            
            if command == 'quit':
                logger.info("Exiting CLI")
                break
            elif command == 'query':
                query = input("Enter your query: ")
                logger.info("Processing query", extra={"query": query})
                response = await controller.execute_query(query)
                print(f"Response: {response}")
            elif command == 'stats':
                stats = await controller.memory_manager.get_memory_stats()
                print("Memory Statistics:")
                for key, value in stats.items():
                    print(f"{key}: {value}")
            elif command == 'distribution':
                distribution = await controller.memory_manager.analyze_link_distribution()
                print("Link Type Distribution:")
                for link_type, count in distribution.items():
                    print(f"{link_type}: {count}")
            elif command == 'visualize':
                await controller.memory_manager.visualize_network()
                print("Network visualization generated and saved as 'memory_network.png'")
            else:
                print("Invalid command. Please try again.")
        except Exception as e:
            logger.error("Error processing command", extra={"command": command, "error": str(e)})
            print(f"An error occurred: {str(e)}. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())