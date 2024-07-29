from openai_client import OpenAIClient
import logging

logger = logging.getLogger(__name__)

class MemoryNeededAgent:
    def __init__(self, model, spawned_controller):
        self.client = OpenAIClient()
        self.spawned_controller = spawned_controller
        self.model = model

    def process_query(self, query):
        try:
            context = self.spawned_controller.gather_context(query)
            response = self.process_with_context(query, context)
            self.spawned_controller.add_interaction(query, response)
            return response
        except Exception as e:
            logger.error(f"Error in Memory Needed Agent: {str(e)}", exc_info=True)
            raise

    def process_with_context(self, query, context):
        context_str = "\n".join([f"Human: {item['prompt']}\nAssistant: {item['response']}" for item in context])
        messages = [
            {"role": "system", "content": "You are an AI assistant with access to previous conversations. Your task is to provide accurate and helpful responses based on the given context and the current query. Always maintain a professional and friendly tone."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nCurrent query: {query}\n\nPlease provide a response that addresses the current query while incorporating relevant information from the context. If the context doesn't contain directly relevant information, use your general knowledge to provide the best possible answer."}
        ]
        return self.client.generate_response(messages, model=self.model)