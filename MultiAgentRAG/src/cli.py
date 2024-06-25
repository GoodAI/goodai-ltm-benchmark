import asyncio
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging

master_logger, chat_logger, memory_logger = setup_logging()
controller = Controller()

async def main():
    while True:
        command = input("Enter a command (query/memories/quit): ").strip().lower()
        
        if command == 'quit':
            break
        elif command == 'query':
            query = input("Enter your query: ")
            response = await controller.execute_query(query)
            print(f"Response: {response}")
        elif command == 'memories':
            limit = int(input("Enter the number of memories to retrieve: "))
            memories = await controller.get_recent_memories(limit)
            for i, (query, result) in enumerate(memories, 1):
                print(f"{i}. Query: {query}\n   Result: {result}\n")
        else:
            print("Invalid command. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())