from dataclasses import dataclass
from typing import Tuple, List
from utils.constants import DATA_DIR
from dataset_interfaces.interface import TestExample
from dataset_interfaces.gpt_generated import GPTGenerated


@dataclass
class DelayedRecallDataset(GPTGenerated):
    name: str = "Delayed Recall"
    description: str = (
        "Give the agent a number of facts about a a fictional world, and then ask 10 questions about these facts."
    )
    generation_file: str = str(DATA_DIR.joinpath("gpt_generation_prompts/1-1_delayed_recall.json"))

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        return 0, 0
