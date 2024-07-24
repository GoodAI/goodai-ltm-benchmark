from utils.together_ai_client import TogetherAIClient

class NMNAgent:
    def __init__(self, model):
        self.client = TogetherAIClient(model)

    def process_query(self, query):
        messages = [{"role": "user", "content": query}]
        return self.client.generate_response(messages)