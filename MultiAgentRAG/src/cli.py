import asyncio
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging
from src.utils.structured_logging import get_logger

master_logger, chat_logger, memory_logger, database_logger = setup_logging()
controller = Controller()
similarity_analyzer = None
logger = get_logger("cli")

async def initialize():
    await controller.initialize()
    global similarity_analyzer

async def main():
    await initialize()
    while True:
        try:
            command = input("Enter a command (query/stats/distribution/visualize/quit): ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'query':
                query = input("Enter your query: ")
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
            print(f"An error occurred: {str(e)}. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())