import logging
from openai_client import OpenAIClient

logger = logging.getLogger(__name__)

class RootController:
    def __init__(self, nmn_agent, memory_needed_agent, model):
        self.nmn_agent = nmn_agent
        self.memory_needed_agent = memory_needed_agent
        self.client = OpenAIClient()
        self.model = model

    def process_query(self, query):
        try:
            if self.requires_memory(query):
                return self.memory_needed_agent.process_query(query)
            else:
                return self.nmn_agent.process_query(query)
        except Exception as e:
            logger.error(f"Error in Root Controller: {str(e)}", exc_info=True)
            raise

    def requires_memory(self, query):
        messages = [
            {"role": "system", "content": "You are an AI tasked with determining if a given query requires access to previous conversation history or additional context to be answered effectively. Respond with 'YES' if the query likely needs additional context, and 'NO' if it can be answered without any prior information. If unsure, default to 'YES'."},
            {"role": "user", "content": f"Query: {query}\n\nDoes this query require access to previous conversation history or additional context to be answered effectively? Respond with 'YES' or 'NO'."}
        ]
        response = self.client.generate_response(messages, model=self.model)
        return response.strip().upper() == "YES"