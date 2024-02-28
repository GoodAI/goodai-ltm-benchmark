from dataclasses import dataclass
from typing import Tuple, List

from dataset_interfaces.interface import DatasetInterface, TestExample

CODED_STATEMENTS = [
    "The house of red is burning today.",
    "The rabbits of Stalingrad run freely this time of year.",
    "The eagle is awakening.",
    "Birds are in the branches.",
    "The blues are in disarray.",
    "Our father is disappointed.",
    "Our mutual friends are waiting for an answer.",
    "What are your colleagues telling you?"
]

@dataclass
class SpyMeetingDataset(DatasetInterface):
    name: str = "Spy Meeting"
    description: str = "Have a cold war esque park bench meeting with a discussion in coded language. Then have the agent try to remember this strange conversation."
    question: str = "What is my favourite colour?"


    def generate_examples(self, num_examples):
        examples = []

        for _ in range(num_examples):
            is_question = []
            script = ["You are on a park bench in a cold January. Someone sits down beside you, opens a newspaper, and starts a conversation with you."]
            num_statements = 3

            coded_copy =
            for stmt_idx in range(num_statements):
                  script.append()




    def evaluate_correct(
            self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        pass

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        pass