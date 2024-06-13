# src/app.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from controller import Controller
from utils.data_utils import load_and_process_data
import os

logging.basicConfig(level=logging.INFO)

def main():
    try:
        # Directly use the saved user variables
        openai_api_key = os.getenv("GOODAI_OPENAI_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")

        if not openai_api_key or not tavily_api_key:
            raise EnvironmentError("API keys not found in environment variables")

        # Set the environment variables for the session
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key

        # Load and process data
        raw_documents = load_and_process_data("data/raw")

        # Create vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(raw_documents, embeddings)

        # Initialize controller
        controller = Controller(vectorstore, "gpt-3.5-turbo", "memory.db")

        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == "quit":
                break

            response = controller.execute_query(query)
            print(f"Response: {response}\n")

            memories = controller.get_memories(5)
            print("Recent Memories:")
            for memory in memories:
                print(f"Query: {memory[0]}")
                print(f"Result: {memory[1]}\n")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
