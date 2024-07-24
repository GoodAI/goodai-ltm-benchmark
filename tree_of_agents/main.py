from config import MAX_TOKENS_PER_AGENT, NMN_MODEL, MEMORY_MODEL
from models.root_controller import RootController
from models.nmn_agent import NMNAgent
from models.memory_needed_agent import MemoryNeededAgent
from controllers.spawned_controller import SpawnedController

def main():
    spawned_controller = SpawnedController(MAX_TOKENS_PER_AGENT)
    nmn_agent = NMNAgent(NMN_MODEL)
    memory_needed_agent = MemoryNeededAgent(MEMORY_MODEL, spawned_controller)
    root_controller = RootController(nmn_agent, memory_needed_agent)

    print("Tree of Agents Prototype")
    print("Enter your queries or type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        response = root_controller.process_query(query)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()