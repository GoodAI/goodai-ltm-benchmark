import json
import os
from datetime import datetime
from config import LEAF_AGENT_DATA_DIR

class LeafAgent:
    def __init__(self, id, max_tokens):
        self.id = id
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.filename = os.path.join(LEAF_AGENT_DATA_DIR, f"leaf_agent_{id}.json")
        self.data = self.load_data()

    def load_data(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        return {"metadata": {}, "interactions": []}

    def save_data(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def add_interaction(self, prompt, response, uid):
        interaction = {
            "uid": uid,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response
        }
        tokens = len(json.dumps(interaction))
        if self.current_tokens + tokens > self.max_tokens:
            return False
        
        self.data["interactions"].append(interaction)
        self.current_tokens += tokens
        self.save_data()
        return True

    def get_relevant_info(self, query):
        relevant_interactions = []
        for interaction in self.data["interactions"]:
            if any(keyword in interaction["prompt"].lower() for keyword in query.lower().split()):
                relevant_interactions.append(interaction)
        return relevant_interactions