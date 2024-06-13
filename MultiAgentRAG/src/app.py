from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from controller import Controller
from utils.data_utils import load_and_process_data

def main():
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

if __name__ == "__main__":
    main()