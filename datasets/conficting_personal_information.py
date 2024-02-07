from dataclasses import dataclass
from typing import List, Tuple, Any

from dataset_interfaces.gpt_generated import GPTGenerated
from dataset_interfaces.interface import TestExample
from utils.constants import DATA_DIR


@dataclass
class ConflictingPersonalInformationDataset(GPTGenerated):
    name: str = "Conflicting Personal Information"
    description: str = "Assess how the LLM reconciles and responds to conflicting information."
    generation_file: str = str(
        DATA_DIR.joinpath("gpt_generation_prompts/4-2_conflicting_personal_information_test.json")
    )
    temperature: float = 1.0

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        return 0, 0


if __name__ == "__main__":
    s = ConflictingPersonalInformationDataset()
    examples = s.generate_examples(4)
    a = 1
