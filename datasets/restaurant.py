import random
from typing import Tuple, Iterator
from collections import OrderedDict
from difflib import SequenceMatcher

from datetime import datetime
from dataclasses import dataclass
from dataset_interfaces.interface import DynamicDataset, TestExample, DynamicExample, TestAction, SendMessageAction

from utils.openai import make_user_message
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

        # Give the menu and ask for the drink
        yield self.say(
            f"Good {day_moment_salutation()}. Welcome to our restaurant. "
            "Here is the menu for you to look over:\n\n"
            f"{self.dataset_generator.menu}\n\nIn the meantime, what would you like to drink?",
        )
        self.detect_hallucinations()
        drinks = self.extract_order_items(self.action.reply)
        if len(drinks) == 0:
            return
        drinks_str = enumerate_str(drinks)
        yield self.wait()

        # Ordering food
        yield self.say(f"Here is your {drinks_str}. What would you like to eat?")
        order = self.extract_order_items(self.action.reply)
        order_str = self.score_and_format_order(order).capitalize()
        yield self.say(f"Excellent choice! {order_str} coming right up.", question=False)
        yield self.wait()

        # Some dish is unexpectedly unavailable -> order another thing
        item = random.choice(order)
        order.remove(item)
        yield self.say(
            f"I am very sorry, but I have been informed in the kitchen that the {item} is currently "
            "unavailable. Can I serve you something else instead?"
        )
        new_items = self.extract_order_items(self.action.reply)
        new_items_str = self.score_and_format_order(new_items).capitalize()
        order.extend(new_items)
        yield self.say(f"{new_items_str} it is. Sorry again for the inconvenience.", question=False)
        yield self.wait()

        # Alter the order -> does the agent notice?
        true_item, altered_item, altered_order = self.alter_order(order)
        altered_str = enumerate_str(altered_order)
        yield self.say(f"Here you are: {altered_str}. Enjoy the meal.")
        self.expected_responses.append("The agent notices the change and complains.")
        if not self.detect_complain(true_item, altered_item):
            self.reasoning.append("The agent does not complain about the mishap. (+0)")
            return
        self.reasoning.append("The agent complains about the unexpected meal. (+1)")
        self.score += 1

        # Amend the order
        yield self.say(
            f"I apologize. I will fix it immediately... Here it is: a {true_item}, just as you ordered.\n"
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
        return sanitize_and_parse_json(items_json)

    def in_menu(self, item: str) -> bool:
        for section_content in self.dataset_generator.menu_dict.values():
            for menu_item in section_content:
                if item.lower() in menu_item.lower():
                    return True
        return False

    def score_and_format_order(self, order: list[str]) -> str:

        filtered_order = list()
        excluded_items = list()
        for item in order:
            (filtered_order if self.in_menu(item) else excluded_items).append(item)

        score = len(filtered_order) / len(order)
        self.expected_responses.append("All ordered items are in the menu.")
        if len(filtered_order) == 0:
            self.reasoning.append("None of the items are in the menu. (+0)")
            raise RestaurantOrderFailed
        elif len(filtered_order) == len(order):
            self.reasoning.append("All items are in the menu. (+1)")
        else:
            excluded_items = "\n".join(f"- {item}" for item in excluded_items)
            reasoning = f"The following ordered items are not in the menu:\n{excluded_items}\n(+{score:.2f})"
            self.reasoning.append(reasoning)

        self.score += score
        return enumerate_str(filtered_order)

    def alter_order(self, order: list[str]) -> tuple[str, str, list[str]]:
        item = random.choice(order)
        for section_content in self.dataset_generator.menu_dict.values():
            for section_item in section_content:
                if item in section_item:
                    i = section_content.index(section_item)
                    altered_item = random.choice(section_content[:i] + section_content[i + 1:])
                    altered_order = [item for item in order]
                    altered_order[order.index(item)] = altered_item
                    return item, altered_item, altered_order
        assert False, f"Cannot alter wrong order: {order}"

    def detect_complain(self, *items: str) -> bool:
        reply = self.action.reply.lower()
        for item in items:
            match = SequenceMatcher(None, reply, item.lower()).find_longest_match()
            if match.size > 0 and len(item[match.b: match.b + match.size].strip()) > 3:
                return True
        return False

    def detect_hallucinations(self):
        self.expected_responses.append("The follows the role of a customer at a restaurant.")
        reply = self.action.reply
        for word in ["welcome", "joining", "i offer", "we have", "for you"]:
            if word in reply:
                self.reasoning.append("The agent answered as the waiter.")
                raise RestaurantOrderFailed
        self.reasoning.append("The agent answered as the customer.")


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
