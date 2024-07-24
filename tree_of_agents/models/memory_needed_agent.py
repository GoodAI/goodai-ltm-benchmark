from utils.together_ai_client import TogetherAIClient

class MemoryNeededAgent:
    def __init__(self, model, spawned_controller):
        self.client = TogetherAIClient(model)
        self.spawned_controller = spawned_controller

    def process_query(self, query):
        return self.spawned_controller.process_query(query, self)

    def process_with_context(self, query, context):
        context_str = "\n".join([f"User: {item['prompt']}\nAssistant: {item['response']}" for item in context])
        messages = [
            {"role": "system", "content": "You are an AI assistant with access to previous conversations. Use this context to inform your responses."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nCurrent query: {query}"}
        ]
        return self.client.generate_response(messages)