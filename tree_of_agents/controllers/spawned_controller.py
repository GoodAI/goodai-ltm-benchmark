from models.leaf_agent import LeafAgent
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)

class SpawnedController:
    def __init__(self, max_tokens_per_agent):
        self.leaf_agents = []
        self.max_tokens_per_agent = max_tokens_per_agent
        self.current_uid = 0

    def gather_context(self, query):
        relevant_context = []
        with ThreadPoolExecutor() as executor:
            future_to_agent = {executor.submit(agent.get_relevant_info, query): agent for agent in self.leaf_agents}
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    if result and result != "NO RELEVANT MEMORIES":
                        relevant_context.extend(result['interactions'])
                except Exception as e:
                    logger.error(f"Error getting context from leaf agent {agent.id}: {str(e)}")
        
        relevant_context.sort(key=lambda x: x['timestamp'], reverse=True)
        return relevant_context

    def add_interaction(self, prompt, response):
        if not self.leaf_agents or not self.leaf_agents[-1].add_interaction(prompt, response, self.current_uid):
            new_agent = LeafAgent(len(self.leaf_agents), self.max_tokens_per_agent)
            new_agent.add_interaction(prompt, response, self.current_uid)
            self.leaf_agents.append(new_agent)
        self.current_uid += 1