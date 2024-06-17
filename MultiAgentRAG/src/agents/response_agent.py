# src/agents/response_agent.py

import logging
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

logger = logging.getLogger('master')

class ResponseAgent:
    def __init__(self, model_name: str):
        self.chat_model = ChatOpenAI(model_name=model_name)

    def generate_response(self, query: str, result: str) -> str:
        messages = [
            HumanMessage(content=f"User Query:\n{query}"),
            AIMessage(content=f"Assistant Response:\n{result}"),
            HumanMessage(content="Generate a final response based on the above interaction.")
        ]
        response = self.chat_model(messages)
        logger.debug(f"Generated final response for query: {query} with result: {result}")
        return response.content
