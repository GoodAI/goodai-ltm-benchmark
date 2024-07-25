from utils.together_ai_client import TogetherAIClient
import logging

logger = logging.getLogger(__name__)

class NMNAgent:
    def __init__(self, model):
        self.client = TogetherAIClient(model)

    def process_query(self, query):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Provide concise and accurate responses."},
                {"role": "user", "content": query}
            ]
            response = self.client.generate_response(messages)
            logger.info(f"NMN Agent processed query successfully")
            return response
        except Exception as e:
            logger.error(f"Error in NMN Agent: {str(e)}", exc_info=True)
            raise