from goodai.ltm.agent import LTMAgent, LTMAgentVariant

from model_interfaces.interface import ChatSession
from utils.constants import PERSISTENCE_DIR, ResetPolicy


class LTMAgentWrapper(ChatSession):
    def __init__(self, model: str, max_prompt_size: int,
                 variant: LTMAgentVariant, run_name: str = ""):
        super().__init__()
        self.model = model
        self.max_prompt_size = max_prompt_size
        self.variant = variant
        self.agent = LTMAgent(variant=variant, model=model, max_prompt_size=max_prompt_size)
        self.costs_usd = 0
        self.run_name = run_name

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size} - {self.variant.name}"

    def reply(self, user_message: str) -> str:
        def _cost_fn(amount: float):
            self.costs_usd += amount

        return self.agent.reply(user_message, cost_callback=_cost_fn)

    def reset(self):
        self.agent.reset()

    def save(self):
        fname = self.save_path.joinpath("full_agent.json")
        with open(fname, "w") as fd:
            fd.write(self.agent.state_as_text())

    def load(self):
        fname = self.save_path.joinpath("full_agent.json")
        with open(fname, "r") as fd:
            state_text = fd.read()
        self.agent.from_state_text(state_text)


