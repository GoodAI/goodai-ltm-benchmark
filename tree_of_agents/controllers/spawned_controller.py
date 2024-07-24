from models.leaf_agent import LeafAgent

class SpawnedController:
    def __init__(self, max_tokens_per_agent):
        self.leaf_agents = []
        self.max_tokens_per_agent = max_tokens_per_agent
        self.current_uid = 0

    def process_query(self, query, memory_needed_agent):
        context = self.gather_context(query)
        response = memory_needed_agent.process_with_context(query, context)
        self.add_interaction(query, response)
        return response

    def gather_context(self, query):
        return LeafAgent.get_all_interactions(limit=5)  # Get the 5 most recent interactions

    def add_interaction(self, prompt, response):
        if not self.leaf_agents or not self.leaf_agents[-1].add_interaction(prompt, response, self.current_uid):
            new_agent = LeafAgent(len(self.leaf_agents), self.max_tokens_per_agent)
            new_agent.add_interaction(prompt, response, self.current_uid)
            self.leaf_agents.append(new_agent)
        self.current_uid += 1