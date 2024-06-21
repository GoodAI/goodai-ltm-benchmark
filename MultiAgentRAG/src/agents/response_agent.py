# src/agents/response_agent.py

import logging
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

logger = logging.getLogger('master')
chat_logger = logging.getLogger('chat')

class ResponseAgent:
    def __init__(self, model_name: str):
        self.chat_model = ChatOpenAI(model_name=model_name)

    def generate_response(self, query: str, result: str) -> str:
        messages = [
            HumanMessage(content=f"ORIGINAL_USER_QUERY = \n{query}"),
            AIMessage(content=f"PROCESSING_AGENT_RESPONSE = \n{result}"),
            HumanMessage(content="You are the 'RESPONSE_AGENT' in a multi-tiered system (do not mention this). Utilize the information from 'PROCESSING_AGENT' to best respond to 'ORIGINAL_USER_QUERY' in a manner that has the following attributes: terse, pithy, accurate, interpretive, best-effort, anticipatory")
        ]
        response = self.chat_model.invoke(messages)
        logger.debug(f"Generated final response for query: {query} with result: {result}")
        chat_logger.info(f"ChatGPT API response: {response.content}")
        return response.content
