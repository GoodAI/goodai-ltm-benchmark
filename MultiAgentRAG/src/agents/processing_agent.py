from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.schema import Document
from typing import List

class ProcessingAgent:
    def __init__(self, model_name: str):
        self.chat_model = ChatOpenAI(model_name=model_name)

    def process(self, query: str, context_documents: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in context_documents])
        messages = [
            HumanMessage(content=f"Given the following context:\n{context}\n\nAnswer the question: {query}")
        ]
        response = self.chat_model(messages)
        return response.content