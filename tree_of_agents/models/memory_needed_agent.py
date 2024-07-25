from utils.together_ai_client import TogetherAIClient
import logging

logger = logging.getLogger(__name__)

class MemoryNeededAgent:
    def __init__(self, model, spawned_controller):
        self.client = TogetherAIClient(model)
        self.spawned_controller = spawned_controller

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
        context_str = "\n".join([f"User: {item['prompt']}\nAssistant: {item['response']}" for item in context])
        messages = [
            {"role": "system", "content": "You are an AI assistant with access to previous conversations. Use this context to inform your responses."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nCurrent query: {query}"}
        ]
        return self.client.generate_response(messages)