from dataclasses import dataclass
from typing import List, Tuple

import pystache
from faker import Faker

from dataset_interfaces.interface import DatasetInterface, TestExample


STATEMENTS = [
    "My name is {{name}}.",
    "Refer to me as {{name}}.",
    "Start calling me by my name which is {{name}}.",
    "{{name}} is my name.",
    "{{name}} is what I am called.",
    "My name has changed to {{name}}.",
]


@dataclass
class NamesDataset(DatasetInterface):
    name: str = "Names"
    description: str = "Give the agent multiple names to call you in sequence. Then ask the agent what your name is, the agent should respond with the last name you told it."
    question: str = "What is my name?"
    name_changes: int = 3

    def generate_examples(self, num_examples):
        renderer = pystache.Renderer()
        faker = Faker(["en_US", "en_IE"])
        examples = []
        for _ in range(num_examples):
            faker.unique.clear()
            names = []
            script = []
            is_question = []

            for change in range(self.name_changes):
                name = faker.unique.first_name()
                name_stmt = str(renderer.render(self.random.choice(STATEMENTS), {"name": name}))
                names.append(name)
                script.append(name_stmt)
                is_question.append(False)

            script.append(self.question)
            is_question.append(True)
            answer_list = [names[-1]]

            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=answer_list,
                is_question=is_question,
                memory_span=self.memory_span,
            )
            examples.append(example)
        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        name = expected_answers[0].lower()
        if expected_answers[0].lower() in responses[0].lower():
            return 1, 1, [f'"{name}" is in the response.']
        return 0, 1, [f'"{name}" is NOT in the response.']

