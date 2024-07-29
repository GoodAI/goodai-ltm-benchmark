from openai_client import OpenAIClient
import logging

logger = logging.getLogger(__name__)

class NMNAgent:
    def __init__(self, model):
        self.client = OpenAIClient()
        self.model = model

    def process_query(self, query):
        try:
            messages = [
                {"role": "system", "content": "You are a highly capable AI assistant designed to provide accurate, concise, and well-reasoned responses. Your task is to answer the user's query to the best of your ability, using logical reasoning and drawing upon your broad knowledge base."},
                {"role": "user", "content": f"Please answer the following query: {query}\n\nProvide a clear and concise response, showing your reasoning where appropriate."}
            ]
            response = self.client.generate_response(messages, model=self.model)
            logger.info(f"NMN Agent processed query successfully")
            return response
        except Exception as e:
            logger.error(f"Error in NMN Agent: {str(e)}", exc_info=True)
            raise