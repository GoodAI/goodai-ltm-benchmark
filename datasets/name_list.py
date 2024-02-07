from random import choice, randint

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
class NameListDataset(DatasetInterface):
    name: str = "NameList"
    description: str = "Give the agent multiple names to call you in sequence. Then ask the agent what all of your names have been. The agent should respond with all of the names you have given it."
    question: str = "What have been all of the names that I have given you?"
    name_changes: int = 3

    def generate_examples(self, num_examples):
        renderer = pystache.Renderer()
        faker = Faker(["en_US", "en_IE"])
        examples = []
        for _ in range(num_examples):
            faker.unique.clear()
            answer_list = []
            script = []
            is_question = []

            for change in range(self.name_changes):
                name = faker.unique.first_name()
                name_stmt = str(renderer.render(choice(STATEMENTS), {"name": name}))
                answer_list.append(name)
                script.append(name_stmt)
                is_question.append(False)

            script.append(self.question)
            is_question.append(True)
            example = TestExample(
                dataset_name=self.name,
                description=self.description,
                dataset_generator=self,
                script=script,
                token_spacings=self.create_filler(is_question),
                expected_responses=answer_list,
                evaluation_fn=self.evaluate_correct,
                number_of_questions=self.count_questions(is_question),
                is_question=is_question,
            )
            examples.append(example)
        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        not_found = []
        for name in expected_answers:
            if not name.lower() in responses[-1].lower():
                not_found.append(name)
        count = len(expected_answers) - len(not_found)
        if len(not_found) == 0:
            reasoning = "All expected names were found in the response."
        else:
            not_found = ", ".join(not_found)
            reasoning = f"{count} names out of {len(expected_answers)} were found. Names {not_found} were not in the response."
        return count, len(expected_answers), [reasoning]

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # All statements are relevant
        # in this test all statements are atomic
        return 0, len(example.script[0])
