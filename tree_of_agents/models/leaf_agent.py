import json
import os
from datetime import datetime
from config import LEAF_AGENT_DATA_DIR, TIMESTAMP_FORMAT
from together import Together
from config import TOGETHER_API_KEY as api_key

class LeafAgent:
    def __init__(self, id, max_tokens):
        self.id = id
        self.max_tokens = max_tokens
        self.filename = os.path.join(LEAF_AGENT_DATA_DIR, f"leaf_agent_{id}.json")
        self.data = self.load_data()
        self.client = self.client = Together(api_key=api_key)

    def load_data(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
        else:
            data = {"metadata": {"current_capacity_tokens": 0}, "interactions": []}
        return data

    def save_data(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def add_interaction(self, prompt, response, uid):
        interaction = {
            "uid": uid,
            "timestamp": datetime.now().strftime(TIMESTAMP_FORMAT),
            "prompt": prompt,
            "response": response
        }
        tokens = len(json.dumps(interaction))
        if self.data["metadata"]["current_capacity_tokens"] + tokens > self.max_tokens:
            return False
        
        self.data["interactions"].append(interaction)
        self.data["metadata"]["current_capacity_tokens"] += tokens
        self.save_data()
        return True

    def get_relevant_info(self, query):
        context = "\n".join([f"Human: {item['prompt']}\nAssistant: {item['response']}" for item in self.data["interactions"]])
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant tasked with determining if the given context contains information relevant to the current query. If there's relevant information, summarize it concisely. If not, respond with 'NO RELEVANT MEMORIES'."},
            {"role": "user", "content": f"Context:\n{context}\n\nCurrent query: {query}\n\nAre there any relevant memories to this query? If so, summarize them. If not, respond with 'NO RELEVANT MEMORIES'."}
        ]
        response = self.client.generate_response(messages)
        
        if response == "NO RELEVANT MEMORIES":
            return response
        else:
            return {
                "interactions": self.data["interactions"],
                "metadata": self.data["metadata"]
            }