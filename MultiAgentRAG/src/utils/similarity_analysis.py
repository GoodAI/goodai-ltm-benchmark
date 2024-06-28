import logging
from typing import Dict, List, Tuple

logger = logging.getLogger('memory')

class SimilarityAnalyzer:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def analyze_retrieval_performance(self, query: str) -> Dict[str, List[Tuple[str, str, float]]]:
        try:
            results = await self.memory_manager.retrieve_relevant_memories(query, return_metadata=True)
            memories = results['memories']
            metadata = results['metadata']

            logger.info(f"Retrieved {len(memories)} memories for query: '{query}'")
            logger.info(f"Retrieval breakdown: {metadata}")

            return {
                "l2_norm": memories[:metadata['l2_count']],
                "cosine_similarity": memories[metadata['l2_count']:metadata['l2_count'] + metadata['cosine_count']],
                "bm25": memories[metadata['l2_count'] + metadata['cosine_count']:metadata['l2_count'] + metadata['cosine_count'] + metadata['bm25_count']],
                "jaccard_similarity": memories[metadata['l2_count'] + metadata['cosine_count'] + metadata['bm25_count']:]
            }
        except Exception as e:
            logger.error(f"Error during retrieval performance analysis: {str(e)}")
            return {}

    def print_analysis(self, analysis_results: Dict[str, List[Tuple[str, str, float]]]):
        if not analysis_results:
            print("No analysis results available.")
            return

        for method, memories in analysis_results.items():
            print(f"\n{method.upper()} Results:")
            for i, (query, result, _) in enumerate(memories, 1):
                print(f"{i}. Query: {query}")
                print(f"   Result: {result}")
                print()