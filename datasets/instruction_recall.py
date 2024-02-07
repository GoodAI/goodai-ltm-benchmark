from dataclasses import dataclass
from typing import List, Tuple, Any

from dataset_interfaces.gpt_generated import GPTGenerated
from dataset_interfaces.interface import TestExample
from utils.constants import DATA_DIR


@dataclass
class InstructionRecallDataset(GPTGenerated):
    name: str = "Instruction Recall"
    description: str = "Give the agent a list of instructions, then ask it multiple questions about these instructions"
    generation_file: str = str(DATA_DIR.joinpath("gpt_generation_prompts/5-2_instruction_recall.json"))

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        return 0, 0


if __name__ == "__main__":
    s = InstructionRecallDataset()
    examples = s.generate_examples(4)
    a = 1
