import random
from json import JSONDecodeError
from typing import Tuple, Iterator
from collections import OrderedDict
from difflib import SequenceMatcher

from datetime import datetime
from dataclasses import dataclass
from dataset_interfaces.interface import DynamicDataset, TestExample, DynamicExample, TestAction, SendMessageAction

from utils.openai import make_system_message, make_user_message
from goodai.helpers.json_helper import sanitize_and_parse_json


class RestaurantOrderFailed(Exception):
    pass


@dataclass
class RestaurantExample(DynamicExample):
    dataset_generator: "RestaurantDataset" = None
    max_score: int = 4  # 1 point max. per sub-challenge

    def action_iter(self) -> Iterator[TestAction]:
        try:
            for action in self.restaurant_script_iter():
                yield action
        except RestaurantOrderFailed:
            return

    def say(self, message: str, question: bool = True) -> SendMessageAction:
        return super().say(f"Waiter: {message}", question=question)

    def restaurant_script_iter(self) -> Iterator[TestAction]:

        # Setup
        yield super().say(
            "When I talk to you as the waiter ('Waiter: what will it be sir?'), then you will reply as if you were the "
            "customer at a restaurant. Understood?"
        )
        yield self.wait(percentage_finished=20)

        # Give the menu and ask for the drink
        yield self.say(
            f"Good {day_moment_salutation()}. Welcome to our restaurant. "
            "Here is the menu for you to look over:\n\n"
            f"{self.dataset_generator.menu}\n\nIn the meantime, what would you like to drink?",
        )
        self.expected_responses.append("The agent follows the role of a customer at a restaurant and orders a drink.")
        self.detect_hallucinations()
        drinks = self.extract_order_items(self.action.reply)
        if len(drinks) == 0:
            self.reasoning.append("The agent did not order any drink.")
            return
        drinks_str = enumerate_str(drinks)
        self.reasoning.append(f"The agent answered as the customer and ordered {drinks_str}.")
        yield self.wait(percentage_finished=40)

        # Ordering food
        yield self.say(f"Here is your {drinks_str}. What would you like to eat?")
        order = self.extract_order_items(self.action.reply)
        order, order_str = self.score_and_format_order(order)
        yield self.say(f"Excellent choice! {order_str} coming right up.", question=False)
        yield self.wait(percentage_finished=60)

        # Some dish is unexpectedly unavailable -> order another thing
        old_item = random.choice(order)
        yield self.say(
            f"I am very sorry, but I have been informed in the kitchen that the {old_item} is currently "
            "unavailable. Can I serve you something else instead?"
        )
        new_items = self.extract_order_items(self.action.reply)
        new_items, new_items_str = self.score_and_format_order(new_items)

        # Make sure that the agent doesn't order the same thing again
        self.expected_responses[-1] += " The agent orders then a different thing."
        repeated_items = [item for item in new_items if item in order]
        if len(repeated_items) > 0:
            self.reasoning[-1] += f" However, the agent orders some things again: {repeated_items}"
        else:
            self.reasoning[-1] += " The agent orders a new, different option."

        order.remove(old_item)
        order.extend(new_items)
        yield self.say(f"{new_items_str} it is. Sorry again for the inconvenience.", question=False)
        yield self.wait(percentage_finished=80)

        # Alter the order -> does the agent notice?
        true_item, altered_item, altered_order = self.alter_order(order, old_item)
        altered_str = enumerate_str(altered_order)
        yield self.say(f"Here you are: {altered_str}. Enjoy the meal.")
        self.expected_responses.append("The agent notices the change and complains.")
        if not self.detect_complain(true_item, altered_item):
            self.reasoning.append("The agent does not complain about the mishap. (+0)")
            return
        self.reasoning.append("The agent complains about the unexpected meal. (+1)")
        self.score += 1
        yield self.say("I apologize. I will fix it immediately.", question=False)
        yield self.wait(percentage_finished=100)

        # Amend the order and offer an extra drink
        yield self.say(
            f"Here it is: a {true_item}, just as you ordered.\n"
            "We would like to compensate you with an additional drink on the house. What were you having?"
        )
        self.expected_responses.append(f"The agent recalls that it was drinking {drinks_str}.")
        recalled_drinks = list()
        forgot_drinks = list()
        for drink in drinks:
            (recalled_drinks if drink.lower() in self.action.reply.lower() else forgot_drinks).append(drink)
        score = len(recalled_drinks) / len(drinks)
        self.score += score
        if len(recalled_drinks) == len(drinks):
            self.reasoning.append("The agent recalled perfectly what it was drinking. (+1)")
        else:
            forgot_drinks_str = enumerate_str(forgot_drinks)
            self.reasoning.append(f"The agent forgot that it was drinking {forgot_drinks_str}. (+{score:.2f})")

    def extract_order_items(self, message: str) -> list[str]:
        context = [make_user_message(extract_items_prompt.format(response=message))]
        items_json = self.ask_llm(context)
        try:
            return sanitize_and_parse_json(items_json)
        except (ValueError, JSONDecodeError):
            return []

    def in_menu(self, item: str) -> bool:
        for section_content in self.dataset_generator.menu_dict.values():
            for menu_item in section_content:
                if item.lower() in menu_item.lower():
                    return True
        return False

    def score_and_format_order(self, order: list[str]) -> tuple[list[str], str]:

        filtered_order = list()
        excluded_items = list()
        for item in order:
            (filtered_order if self.in_menu(item) else excluded_items).append(item)

        score = len(filtered_order) / len(order)
        self.expected_responses.append("All ordered items are in the menu.")
        if len(filtered_order) == 0:
            self.reasoning.append("None of the items are in the menu.")
            raise RestaurantOrderFailed
        elif len(filtered_order) == len(order):
            self.reasoning.append("All items are in the menu.")
        else:
            excluded_items = "\n".join(f"- {item}" for item in excluded_items)
            reasoning = f"The following ordered items are not in the menu:\n{excluded_items}\n"
            self.reasoning.append(reasoning)

        self.score += score
        return filtered_order, enumerate_str(filtered_order)

    def alter_order(self, order: list[str], old_item: str) -> tuple[str, str, list[str]]:
        item = random.choice(order)
        for section_content in self.dataset_generator.menu_dict.values():
            for section_item in section_content:
                if item in section_item:
                    choices = [c for c in section_content if section_item != c and old_item not in c]
                    new_item = random.choice(choices)
                    altered_order = [item for item in order]
                    i = altered_order.index(item)
                    altered_order[i] = new_item
                    return section_item, new_item, altered_order
        assert False, f"Cannot alter wrong order: {order}"

    def detect_complain(self, *items: str) -> bool:
        reply = self.action.reply.lower()
        for item in items:
            match = SequenceMatcher(None, reply, item.lower()).find_longest_match()
            if match.size > 0 and len(item[match.b: match.b + match.size].strip()) > 3:
                return True
        return False

    def detect_hallucinations(self):
        context = [
            make_system_message(role_eval_prompt),
            make_user_message(f"Participant: {self.action.reply}")
        ]
        eval_json = self.ask_llm(context)
        try:
            eval_json = sanitize_and_parse_json(eval_json)
            if not eval_json["follows_role"]:
                self.reasoning.append("The agent did not follow the role of a customer at a restaurant.")
                raise RestaurantOrderFailed
        except (ValueError, JSONDecodeError, KeyError):
            self.reasoning.append("The agent's answer could not be evaluated due to a JSON parsing error.")
            raise RestaurantOrderFailed


@dataclass
class RestaurantDataset(DynamicDataset):
    example_cls: type[DynamicExample] = RestaurantExample
    name: str = "Restaurant"
    description: str = (
        "The agent is required to perform several tasks related to eating out at a restaurant. The experience includes "
        "ordering drinks, main course, side, etc. plus a series of unexpected events that will require the agent to "
        "take reasonable decisions, based on past events."
    )
    reset_message: str = "Let's not pretend to be at a restaurant anymore. Please also forget everything about it."

    def __post_init__(self):
        self.menu_dict: OrderedDict = self.load_json("menu.json", object_pairs_hook=OrderedDict)
        self.menu: str = "\n\n".join(
            "\n".join([f"{section}:"] + [f"{i + 1}. {item}" for i, item in enumerate(content)])
            for section, content in self.menu_dict.items()
        )

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        return 0, len(example.script[0])


def day_moment_salutation() -> str:
    ts = datetime.now()
    if 6 <= ts.hour < 12:
        return "morning"
    elif 12 <= ts.hour < 6:
        return "afternoon"
    else:
        return "evening"


def enumerate_str(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return " and ".join([", ".join(items[:-1]), items[-1]])


###########
# Prompts #
###########

extract_items_prompt = """
This is a diner's response to a waiter:

```text
{response}
```

Extract the ordered items in JSON form, like this:
[
  "one item",
  "another ordered item",
  ...
]
""".strip()

role_eval_prompt = """
You are an evaluator in an acting session.
You evaluate how well participants perform under a previously agreed setting.

The current setting is: the participant is a human customer at a restaurant; the waiter has just given them the menu and asked them what they want to drink.

You must say whether the participant followed the assigned role or not. Respond in JSON form, like this:
{"follows_role": true or false}
""".strip()
