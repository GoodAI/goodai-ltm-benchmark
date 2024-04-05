import logging
from json import JSONDecodeError

import pystache
from dataclasses import dataclass
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
                if len(cart) > 0 and self.random.randint(1, 6) < 2:
                    # remove
                    item = self.random.choice(cart)
                    current_number = counts[cart.index(item)]
                    number = self.random.randint(1, current_number)
                    current_number -= number
                    if current_number <= 0:
                        del counts[cart.index(item)]
                        cart.remove(item)
                    else:
                        counts[cart.index(item)] = current_number
                    statement = renderer.render(
                        self.random.choice(STATEMENTS_REMOVE), {"item": item, "number": str(number)}
                    )
                else:
                    # add
                    item = self.random.choice(ITEMS)
                    number = self.random.choice(NUMBER)
                    modifier = ""
                    if item in cart:
                        modifier = "another "
                        counts[cart.index(item)] += number
                    else:
                        cart.append(item)
                        counts.append(number)
                    statement = renderer.render(
                        self.random.choice(STATEMENTS_ADD),
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
                expected_responses=answer_list,
                is_question=is_question,
                memory_span=self.memory_span,
            )

            examples.append(example)

        return examples

    def evaluate_correct(
        self,
        questions: List[str],
        responses: List[str],
        expected_answers: List[Tuple[str, int]],
    ) -> Tuple[float, int, List[str]]:
        """
        The evaluation will give up to `item_changes` points in total.
        That punctuation can be broken down in thirds:
        1. One third depends on the number of expected items present.
        2. Another third depends on those items' quantities matching the expected values.
        3. The last third is given if there are no hallucinated items.
        """
        score = 0
        max_score = self.item_changes
        num_correct = 0
        real_items = []
        hallucinated_items = []
        reasoning = []

        expected_names = []
        expected_items = {}
        for a in expected_answers:
            expected_names.append(a[0])
            expected_items[a[0]] = a[1]

        # Check response format
        try:
            answer_items = sanitize_and_parse_json(responses[0])
            assert isinstance(answer_items, list)
            for item in answer_items:
                assert isinstance(item["item"], str)
                assert isinstance(item["quantity"], int)
        except (JSONDecodeError, ValueError, KeyError, AssertionError) as exc:
            msg = f"Response not in correct format ({repr(exc)}):\n{responses[0]}"
            logging.exception(msg)
            return score, max_score, [msg]

        # Evaluate
        for item in answer_items:
            name = item["item"].lower()
            matched, key = match_plural(name, expected_names)
            if not matched:
                hallucinated_items.append(name)
                continue
            real_items.append(name)
            if item["quantity"] == expected_items[key]:
                num_correct += 1
            else:
                reasoning.append(f"Wrong quantity for {name}: {item['quantity']} vs {expected_items[key]}.")

        score += len(real_items) / len(expected_answers)
        if len(real_items) < len(expected_answers):
            reasoning.append(f"{len(expected_answers) - len(real_items)} items were not found in the response:")
            reasoning.extend(f"- {name}" for name in expected_names if name not in real_items)

        score += num_correct / len(expected_answers)
        if num_correct == len(expected_answers):
            reasoning.append("All expected items' quantities match.")

        score += int(hallucinated_items == [])
        if len(hallucinated_items) > 0:
            reasoning.append(f"{len(hallucinated_items)} unexpected items were found:")
            reasoning.extend(f"- {name}" for name in hallucinated_items)

        score = (score / 3) * max_score
        return score, max_score, ["\n".join(reasoning)]

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
