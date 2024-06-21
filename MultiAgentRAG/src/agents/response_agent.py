# src/agents/response_agent.py

import logging
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

logger = logging.getLogger('master')

class ResponseAgent:
    def __init__(self, model_name: str):
        self.chat_model = ChatOpenAI(model_name=model_name)

    def generate_response(self, query: str, result: str) -> str:
        messages = [
            HumanMessage(content=f"ORIGINAL_USER_QUERY = \n{query}"),
            AIMessage(content=f"PROCESSING_AGENT_RESPONSE = \n{result}"),
            HumanMessage(content="You are the 'RESPONSE_AGENT' in a multi tiered system. Utilize the information from 'PROCESSING_AGENT' to best respond to ORIGINAL_USER_QUERY")
        ]
        response = self.chat_model.invoke(messages)
        logger.debug(f"Generated final response for query: {query} with result: {result}")
        return response.content
