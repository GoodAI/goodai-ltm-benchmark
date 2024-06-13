from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class RetrievalAgent:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query)