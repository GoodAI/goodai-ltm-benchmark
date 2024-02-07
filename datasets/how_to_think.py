from dataclasses import dataclass
from typing import List, Tuple, Any


from dataset_interfaces.gpt_generated import GPTGenerated

from dataset_interfaces.interface import TestExample
from utils.constants import DATA_DIR


@dataclass
class HowToThinkDataset(GPTGenerated):
    name: str = "How to Think"
    description: str = (
        "Evaluate the LLM's ability to apply a given reasoning or thought process to a particular situation or problem."
    )
    generation_file: str = str(DATA_DIR.joinpath("gpt_generation_prompts/7-1_teaching_how_to_think.json"))
    temperature: float = 1.0

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        return 0, 0


if __name__ == "__main__":
    s = HowToThinkDataset()
    examples = s.generate_examples(4)
    a = 1
