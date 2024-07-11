import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

def visualize_memory_network(memories: List[Tuple[int, str, str]], links: List[Tuple[int, int, str]]):
    G = nx.Graph()
    
    # Add nodes
    for memory_id, query, _ in memories:
        G.add_node(memory_id, label=query[:20])  # Use first 20 characters of query as label
    
    # Add edges
    for source_id, target_id, link_type in links:
        G.add_edge(source_id, target_id, label=link_type)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, {node: data['label'] for node, data in G.nodes(data=True)})
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Memory Network Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('memory_network.png')
    plt.close()

# Usage in MemoryManager:
# async def visualize_network(self):
#     memories = await self.get_all_memories()
#     links = await self.get_all_links()
#     visualize_memory_network([(m['id'], m['query'], m['result']) for m in memories], links)