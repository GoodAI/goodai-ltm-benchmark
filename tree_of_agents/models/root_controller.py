class RootController:
    def __init__(self, nmn_agent, memory_needed_agent):
        self.nmn_agent = nmn_agent
        self.memory_needed_agent = memory_needed_agent

    def process_query(self, query):
        if self.requires_memory(query):
            return self.memory_needed_agent.process_query(query)
        else:
            return self.nmn_agent.process_query(query)

    def requires_memory(self, query):
        memory_keywords = ['history', 'previous', 'context', 'remember']
        return any(keyword in query.lower() for keyword in memory_keywords)