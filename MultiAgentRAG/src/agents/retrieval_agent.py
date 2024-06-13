from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List

class RetrievalAgent:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query)