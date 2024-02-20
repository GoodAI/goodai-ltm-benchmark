import codecs
import json
import os
import threading

from goodai.ltm.agent import LTMAgent, LTMAgentVariant
from model_interfaces.interface import ChatSession

_log_prompts = os.environ.get("LTM_BENCH_PROMPT_LOGGING", "False").lower() in ["true", "yes", "1"]


class LTMAgentWrapper(ChatSession):
    def __init__(self, model: str, max_prompt_size: int,
                 variant: LTMAgentVariant):
        super().__init__()
        self.model = model
        self.max_prompt_size = max_prompt_size
        self.variant = variant
        self.log_lock = threading.RLock()
        self.log_count = 0
        self.agent = LTMAgent(variant=variant, model=model, max_prompt_size=max_prompt_size,
                              prompt_callback=self._prompt_callback)
        self.costs_usd = 0

    def _prompt_callback(self, session_id: str, label: str, context: list[dict], completion: str):
        if _log_prompts:
            with self.log_lock:
                self.log_count += 1
                log_dir = f"./logs/{session_id}"
                os.makedirs(log_dir, exist_ok=True)
                prompt_file = f"{label}-prompt-{self.log_count}.json"
                prompt_json = json.dumps(context, indent=2)
                prompt_path = os.path.join(log_dir, prompt_file)
                with codecs.open(prompt_path, "w", "utf-8") as fd:
                    fd.write(prompt_json)
                completion_path = os.path.join(log_dir, f"{label}-completion-{self.log_count}.txt")
                with codecs.open(completion_path, "w", "utf-8") as fd:
                    fd.write(completion)

    @property
    def name(self):
        return f"{super().name} - {self.model} - {self.max_prompt_size} - {self.variant.name}"

    def reply(self, user_message: str) -> str:
        def _cost_fn(amount: float):
            self.costs_usd += amount

        return self.agent.reply(user_message, cost_callback=_cost_fn)

    def reset(self):
        self.agent.reset()
