from goodai.ltm.agent import LTMAgent, LTMAgentVariant
from model_interfaces.interface import ChatSession


class LTMAgentWrapper(ChatSession):
    def __init__(self, model: str, max_prompt_size: int,
                 variant: LTMAgentVariant):
        super().__init__()
        self.model = model
        self.max_prompt_size = max_prompt_size
        self.variant = variant
        self.agent = LTMAgent(variant=variant, model=model, max_prompt_size=max_prompt_size)
        self.costs_usd = 0

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size} - {self.variant.name}"

    def reply(self, user_message: str) -> str:
        def _cost_fn(amount: float):
            self.costs_usd += amount

        return self.agent.reply(user_message, cost_callback=_cost_fn)

    def reset(self):
        self.agent.reset()
