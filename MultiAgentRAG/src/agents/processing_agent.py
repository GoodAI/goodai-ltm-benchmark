# src/agents/processing_agent.py

import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, Document
from typing import List

logger = logging.getLogger('master')

class ProcessingAgent:
    def __init__(self, model_name: str):
        self.chat_model = ChatOpenAI(model_name=model_name)

    def process(self, query: str, context_documents: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in context_documents[:3]])
        messages = [
            HumanMessage(content=f"Given the following context:\n{context}\n\nAnswer the question: {query}")
        ]
        response = self.chat_model.invoke(messages)
        logger.debug(f"Processed query: {query} with context: {context}")
        return response.content

