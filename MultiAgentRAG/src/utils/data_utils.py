# src/utils/data_utils.py

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger('master')

def structure_memories(memories: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Transforms a list of memory tuples into a structured list of dictionaries.

    Args:
        memories (List[Tuple[str, str]]): List of memories where each memory is a tuple (query, result).

    Returns:
        List[Dict[str, str]]: List of structured memories with titles, queries, results, and tags.
    """
    structured_memories = []
    for idx, (query, result) in enumerate(memories):
        memory_data = {
            "title": f"Memory_{idx + 1}",
            "query": query,
            "result": result,
            "tags": "retrieved, processed"
        }
        structured_memories.append(memory_data)
    return structured_memories
