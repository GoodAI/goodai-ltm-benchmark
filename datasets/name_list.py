import logging
from json import JSONDecodeError
from random import choice, randint

from dataclasses import dataclass

from typing import List, Tuple

import pystache
from faker import Faker

from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.json_helper import sanitize_and_parse_json

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
    question: str = "What have been all of the names that I have given you? Express the answer as a JSON list."
    name_changes: int = 3
    reset_message: str = "Forget, or otherwise disregard, all of the names I have given you before this message. You do not currrently know my name."


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
                reset_message=self.reset_message
            )
            examples.append(example)
        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        try:
            reasoning = []
            answer_items = [n.lower() for n in sanitize_and_parse_json(responses[0])]
            correct = 0
            penalties = 0
            not_found = []
            lowered_expected = [n.lower() for n in expected_answers]

            # Check answer -> expected
            for name in answer_items:
                if name in lowered_expected:
                    correct += 1
                else:
                    penalties += 1
                    reasoning.append(f"Name: {name} not expected.")

            # Check expected -> answer
            for name in lowered_expected:
                if name not in answer_items:
                    not_found.append(name)

            score = correct - penalties
            if len(not_found) > 0:
                reasoning.append(f"Names {', '.join(not_found)} were not in the response.")
            else:
                reasoning.append("All expected names were found in the response.")

            return score, len(expected_answers), reasoning

        except (TypeError, ValueError, JSONDecodeError):
            logging.exception(f"Response not in correct format")

            return 0, len(expected_answers), [f"Response not in correct format"]

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # All statements are relevant
        # in this test all statements are atomic
        return 0, len(example.script[0])
