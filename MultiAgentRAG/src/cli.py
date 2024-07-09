import asyncio
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging

master_logger, chat_logger, memory_logger, database_logger = setup_logging()
controller = Controller()
similarity_analyzer = None

async def initialize():
    await controller.initialize()
    global similarity_analyzer

async def main():
    await initialize()
    while True:
        try:
            command = input("Enter a command (query/quit): ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'query':
                query = input("Enter your query: ")
                response = await controller.execute_query(query)
                print(f"Response: {response}")
            else:
                print("Invalid command. Please try again.")
        except Exception as e:
            master_logger.error(f"An error occurred: {str(e)}", exc_info=True)
            print(f"An error occurred: {str(e)}. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())