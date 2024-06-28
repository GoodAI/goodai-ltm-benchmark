import asyncio
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging
from src.utils.similarity_analysis import SimilarityAnalyzer

master_logger, chat_logger, memory_logger, database_logger = setup_logging()
controller = Controller()
similarity_analyzer = None

async def initialize():
    await controller.initialize()
    global similarity_analyzer
    similarity_analyzer = SimilarityAnalyzer(controller.memory_manager)

async def main():
    await initialize()
    while True:
        try:
            command = input("Enter a command (query/memories/consistency/analyze/quit): ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'query':
                query = input("Enter your query: ")
                response = await controller.execute_query(query)
                print(f"Response: {response}")
            elif command == 'memories':
                try:
                    limit = int(input("Enter the number of memories to retrieve: "))
                    memories = await controller.get_recent_memories(limit)
                    for i, (query, result) in enumerate(memories, 1):
                        print(f"{i}. Query: {query}\n   Result: {result}\n")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif command == 'consistency':
                await controller.memory_manager.run_consistency_check_and_fix()
                print("Consistency check and fix completed.")
            elif command == 'analyze':
                query = input("Enter a query to analyze: ")
                analysis_results = await similarity_analyzer.analyze_retrieval_performance(query)
                similarity_analyzer.print_analysis(analysis_results)
            else:
                print("Invalid command. Please try again.")
        except Exception as e:
            master_logger.error(f"An error occurred: {str(e)}", exc_info=True)
            print(f"An error occurred: {str(e)}. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())