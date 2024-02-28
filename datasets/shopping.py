import logging
from json import JSONDecodeError

import pystache
from dataclasses import dataclass
from random import randint, choice
from typing import List, Tuple

from goodai.helpers.json_helper import sanitize_and_parse_json

from dataset_interfaces.interface import DatasetInterface, TestExample

ITEMS = [
    "Bread",
    "Milk",
    "Cheese",
    "Carrot",
    "Chicken",
    "Toilet Paper",
    "Toothpaste",
    "Soap",
    "Leek",
    "Celery",
    "Steak",
    "Potato",
]

STATEMENTS_ADD = [
    "Please add {{modifier}}{{number}} {{item}} to my shopping list",
    "I require {{modifier}}{{number}} {{item}}, please put it on the list.",
    # "If I am not currently planning on getting {{item}}, please plan it along with the others.",
    "I want to buy {{modifier}}{{number}} {{item}}.",
]

STATEMENTS_REMOVE = [
    # "I already have {{item}}, remove it from the list.",
    "Remove {{number}} {{item}}.",
    "Please remove {{number}} {{item}} from my shopping list"
    # "Please remove all occurrences of {{item}} from my shopping list.",
]

NUMBER = [1, 2, 3]


def match_plural(answer_name: str, expected_names: List[str]):
    for name in expected_names:
        # The answer_name could be the one that is plural
        if name in answer_name:
            return True, name

    return False, ""


@dataclass
class ShoppingDataset(DatasetInterface):
    name: str = "Shopping"
    description: str = "Give the agent multiple statements adding and removing items from a shopping list. Then ask the agent what is on the shopping list."
    question: str = "What is on my current shopping list? Express the list of items as a JSON list of objects with `item` and `quantity` properties only. Consolidate items that are the same."
    item_changes: int = 3
    reset_message: str = "I have bought all of the items on the list. Please remove all of the items on the current shopping list."

    def generate_examples(self, num_examples):
        renderer = pystache.Renderer()
        examples = []
        for _ in range(num_examples):
            counts = []
            cart = []
            script = []
            is_question = []

            for change in range(self.item_changes):
                if len(cart) > 0 and randint(1, 6) < 2:
                    # remove
                    item = choice(cart)
                    current_number = counts[cart.index(item)]
                    number = randint(1, current_number)
                    current_number -= number
                    if current_number <= 0:
                        del counts[cart.index(item)]
                        cart.remove(item)
                    else:
                        counts[cart.index(item)] = current_number
                    statement = renderer.render(
                        choice(STATEMENTS_REMOVE), {"item": item, "number": str(number)}
                    )
                else:
                    # add
                    item = choice(ITEMS)
                    number = choice(NUMBER)
                    modifier = ""
                    if item in cart:
                        modifier = "another "
                        counts[cart.index(item)] += number
                    else:
                        cart.append(item)
                        counts.append(number)
                    statement = renderer.render(
                        choice(STATEMENTS_ADD),
                        {"item": item, "modifier": modifier, "number": str(number)},
                    )

                script.append(statement)
                is_question.append(False)

            script.append(self.question)
            is_question.append(True)

            answer_list = []
            for co, it in zip(counts, cart):
                answer_list.append((it.lower(), co))

            example = TestExample(
                dataset_generator=self,
                script=script,
                token_spacings=self.create_filler(is_question),
                expected_responses=answer_list,
                is_question=is_question,
            )

            examples.append(example)

        return examples

    def evaluate_correct(
        self,
        questions: List[str],
        responses: List[str],
        expected_answers: List[Tuple[str, int]],
    ) -> Tuple[int, int, List[str]]:
        max_score = len(expected_answers)
        num_correct = 0
        penalties = 0
        errors = []
        try:
            answer_items = sanitize_and_parse_json(responses[0])
            expected_items = {}
            expected_names = []
            for a in expected_answers:
                expected_names.append(a[0])
                expected_items[a[0]] = a[1]
            for item in answer_items:
                name = item["item"].lower()
                matched, key = match_plural(name, expected_names)
                if matched:
                    if item["quantity"] == expected_items[key]:
                        num_correct += 1
                    else:
                        errors.append(f"Wrong quantity for {name}: {item['quantity']} vs {expected_items[key]}.")
                else:
                    penalties += 1
                    errors.append(f'Unknown item {name}.')
        except (TypeError, ValueError, JSONDecodeError):
            logging.exception(f"Response not in correct format")
            num_correct = 0
        if num_correct < max_score:
            errors.append(f"{max_score - num_correct} items were not found in the response.")

        num_correct = max(num_correct - penalties, 0)
        if num_correct == max_score:
            reasoning = "All items and quantities match."
        else:
            reasoning = "\n".join(errors)
        return num_correct, max_score, [reasoning]

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # All statements are relevant
        # in this test all statements are atomic
        return 0, len(example.script[0])


def main():
    # Create a conversation for the agent using a name and phil
    items = ShoppingDataset()
    items.generate_examples(1)


if __name__ == "__main__":
    main()
