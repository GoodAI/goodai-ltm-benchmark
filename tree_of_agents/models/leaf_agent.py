import json
from datetime import datetime

class LeafAgent:
    def __init__(self, id, max_tokens):
        self.id = id
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.content = []
        self.filename = f"data/leaf_agents/leaf_agent_{id}.json"
        self.load_content()

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
        self.content.append(interaction)
        self.current_tokens += tokens
        self.save_content()
        return True

    def load_content(self):
        try:
            with open(self.filename, 'r') as f:
                self.content = json.load(f)
                self.current_tokens = len(json.dumps(self.content))
        except FileNotFoundError:import json
import os
from datetime import datetime

class LeafAgent:
    def __init__(self, id, max_tokens):
        self.id = id
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.content = []
        self.directory = "data/leaf_agents"
        self.filename = f"{self.directory}/leaf_agent_{id}.json"
        os.makedirs(self.directory, exist_ok=True)
        self.load_content()

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
        self.content.append(interaction)
        self.current_tokens += tokens
        self.save_content()
        return True

    def load_content(self):
        try:
            with open(self.filename, 'r') as f:
                self.content = json.load(f)
                self.current_tokens = sum(len(json.dumps(item)) for item in self.content)
        except FileNotFoundError:
            self.content = []
            self.current_tokens = 0

    def save_content(self):
        with open(self.filename, 'w') as f:
            json.dump(self.content, f)

    @classmethod
    def get_all_interactions(cls, limit=None):
        all_interactions = []
        for filename in os.listdir(cls.directory):
            if filename.startswith("leaf_agent_") and filename.endswith(".json"):
                with open(os.path.join(cls.directory, filename), 'r') as f:
                    interactions = json.load(f)
                    all_interactions.extend(interactions)
        
        all_interactions.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_interactions[:limit] if limit else all_interactions

    def save_content(self):
        with open(self.filename, 'w') as f:
            json.dump(self.content, f)