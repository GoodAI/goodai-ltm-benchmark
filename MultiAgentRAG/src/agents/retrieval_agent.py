# src/agents/retrieval_agent.py

import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List

logger = logging.getLogger('master')

class RetrievalAgent:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(self, query: str) -> List[Document]:
        results = self.vectorstore.similarity_search(query)
        logger.debug(f"Retrieved documents for query: {query}")
        return results
