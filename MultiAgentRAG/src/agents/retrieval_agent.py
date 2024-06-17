# src/agents/retrieval_agent.py

import logging
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List

logger = logging.getLogger('master')

class RetrievalAgent:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(self, query: str) -> List[Document]:
        results = self.vectorstore.similarity_search(query)
        # Rank documents based on relevance score
        ranked_results = sorted(results, key=lambda doc: doc.metadata.get('relevance_score', 0), reverse=True)
        logger.debug(f"Retrieved and ranked documents for query: {query}")
        return ranked_results
