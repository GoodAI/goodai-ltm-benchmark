# src/agents/processing_agent.py

import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, Document
from typing import List

logger = logging.getLogger('master')
chat_logger = logging.getLogger('chat')

class ProcessingAgent:
    def __init__(self, model_name: str):
        self.chat_model = ChatOpenAI(model_name=model_name)

    def process(self, query: str, context_documents: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in context_documents[:3]])
        messages = [
            HumanMessage(content=f"REQUEST = \"{query}\". CONTEXT = \"\"\"{context}\"\"\" INSTRUCTIONS = Your context is formed of reverse chronologically ordered 'memories' in the format <REQUEST>,<RESPONSE>. You will use these to best respond to the request. 'Best' is defined as the response that will satisfy the sender of the request the most.")
        ]
        response = self.chat_model.invoke(messages)
        logger.debug(f"Processed query: {query} with context: {context}")
        chat_logger.info(f"ChatGPT API response: {response.content}")
        return response.content
