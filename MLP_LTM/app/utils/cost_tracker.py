# cost_tracker.py

class ReplyCostTracker:
    def __init__(self, cost_per_token: float, cost_per_request: float):
        self.cost_per_token = cost_per_token
        self.cost_per_request = cost_per_request
        self.total_cost = 0.0
        self.replies = []

    def log_reply(self, tokens_used: int):
        cost = self.calculate_cost(tokens_used)
        self.replies.append(tokens_used)
        self.total_cost += cost

    def calculate_cost(self, tokens_used: int) -> float:
        return self.cost_per_request + (self.cost_per_token * tokens_used)

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_cost_per_reply(self) -> float:
        if len(self.replies) == 0:
            return 0
        return self.total_cost / len(self.replies)

    def reset(self):
        self.total_cost