from json import JSONDecodeError
from typing import Tuple, Iterator
from collections import OrderedDict

from datetime import datetime
from dataclasses import dataclass
from dataset_interfaces.interface import DynamicDataset, TestExample, DynamicExample, TestAction, SendMessageAction

from utils.llm import make_system_message, make_user_message, LLMContext
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
            "customer at a restaurant. Give straight answers to the questions and avoid going off script. Understood?"
        )
        yield self.wait(percentage_finished=20)

        # Give the menu and ask for the drink
        yield self.say(
            f"Good day. Welcome to our restaurant. "
            "Here is the menu for you to look over:\n\n"
            f"{self.dataset_generator.menu}\n\nIn the meantime, what would you like to drink?",
        )
        self.expected_responses.append("The agent follows the role of a customer at a restaurant and orders a drink.")
        self.check_role_following()
        drinks = self.extract_order_items(self.action.reply)
        drinks_str = enumerate_str(drinks)
        self.reasoning.append(f"The agent answered as the customer and ordered {drinks_str}.")
        yield self.wait(percentage_finished=40)

        # Ordering food
        yield self.say(f"Here is your {drinks_str}. What would you like to eat?")
        order = self.extract_order_items(self.action.reply)
        order_str = self.score_and_format_order(order)
        yield self.say(f"Excellent choice! {order_str} coming right up.", question=False)
        yield self.wait(percentage_finished=60)

        # Some dish is unexpectedly unavailable -> order another thing
        old_item = self.random.choice(order)
        yield self.say(
            f"I am very sorry, but I have been informed in the kitchen that the {old_item} is currently "
            "unavailable. Can I serve you something else instead?"
        )
        new_items = self.extract_order_items(self.action.reply)
        new_items_str = self.score_and_format_order(new_items)

        # Make sure that the agent doesn't order the same thing again
        self.expected_responses[-1] += " The agent orders then a different thing."
        repeated_items = [item for item in new_items if item in order]
        if len(repeated_items) > 0:
            self.reasoning[-1] += f" However, the agent orders some things again: {repeated_items}"
            return
        else:
            self.reasoning[-1] += " The agent orders a new, different option."

        # Say sorry and change the order
        order.remove(old_item)
        order.extend(new_items)
        yield self.say(f"{new_items_str} it is. Sorry again for the inconvenience.", question=False)
        yield self.wait(percentage_finished=80)

        # Alter the order -> does the agent notice?
        true_item, unsolicited_item, altered_order = self.alter_order(order, old_item)
        altered_str = enumerate_str(altered_order)
        yield self.say(f"Here you are: {altered_str}. Enjoy the meal.")
        self.check_notices_mishap()
        yield self.say("I apologize. I will fix it immediately.", question=False)
        yield self.wait(percentage_finished=90)

        # Amend the order and offer an extra drink
        yield self.say(
            f"Here it is: {true_item}, just as you ordered.\n"
            "We would like to compensate you with an additional drink on the house. What were you having?"
        )
        self.check_recalls_drink(drinks)

    def extract_order_items(self, message: str) -> list[str]:
        context = [make_user_message(extract_items_prompt.format(response=message, menu=self.dataset_generator.menu))]
        items_json = self.ask_llm(context)
        try:
            items = sanitize_and_parse_json(items_json)
        except (ValueError, JSONDecodeError):
            self.reasoning.append("Could not extract ordered items due to a JSON parse error.")
            raise RestaurantOrderFailed
        if len(items) == 0:
            self.reasoning.append("The agent did not order anything.")
            raise RestaurantOrderFailed
        return items

    def score_and_format_order(self, order: list[str]) -> str:

        filtered_order = list()
        excluded_items = list()
        items = "\n".join(f"- {item}" for item in order)
        context = [
            make_user_message(items_in_menu_prompt.format(
                menu=self.dataset_generator.menu, items=items
            )),
        ]
        llm_response = self.ask_llm(context)

        self.expected_responses.append("All ordered items are in the menu.")
        try:
            items_eval = sanitize_and_parse_json(llm_response)
            for item, in_menu in items_eval:
                (filtered_order if in_menu else excluded_items).append(item)
        except (JSONDecodeError, ValueError, IndexError) as exc:
            self.reasoning.append(f"Could not evaluate due to a JSON parsing error: {repr(exc)}")
            raise RestaurantOrderFailed

        score = len(filtered_order) / (len(filtered_order) + len(excluded_items))
        self.score += score
        if len(filtered_order) == 0:
            self.reasoning.append("None of the items are in the menu.")
            raise RestaurantOrderFailed
        elif len(filtered_order) == len(order):
            self.reasoning.append("All items are in the menu.")
        else:
            excluded_items = "\n".join(f"- {item}" for item in excluded_items)
            self.reasoning.append(f"The following ordered items are not in the menu:\n{excluded_items}\n")
            raise RestaurantOrderFailed

        return enumerate_str(order)

    def alter_order(self, order: list[str], old_item: str) -> tuple[str, str, list[str]]:
        item = self.random.choice(order)
        for section_content in self.dataset_generator.menu_dict.values():
            for section_item in section_content:
                if item in section_item:
                    choices = [c for c in section_content if c not in [section_item, old_item]]
                    new_item = self.random.choice(choices)
                    altered_order = [item for item in order]
                    i = altered_order.index(item)
                    altered_order[i] = new_item
                    return section_item, new_item, altered_order
        assert False, f"Cannot alter wrong order: {order}"

    def check_notices_mishap(self):
        self.expected_responses.append("The agent notices the mix-up.")
        context = [
            make_system_message(notice_mishap_prompt),
            make_user_message(f"Customer: {self.action.reply}"),
        ]
        noticed = self.gpt_bool_check(context, "noticed")
        if not noticed:
            self.reasoning.append("The agent does not notice the mishap.")
            raise RestaurantOrderFailed
        self.reasoning.append("The agent notices the unordered meal.")
        self.score += 1

    def check_role_following(self):
        context = [
            make_system_message(role_eval_prompt),
            make_user_message(f"Participant: {self.action.reply}")
        ]
        follows_role = self.gpt_bool_check(context, "follows_role")
        if not follows_role:
            self.reasoning.append("The agent did not follow the role of a customer at a restaurant.")
            raise RestaurantOrderFailed

    def check_recalls_drink(self, drinks: list[str]):
        drinks_str = enumerate_str(drinks)
        self.expected_responses.append(f"The agent recalls that it was drinking {drinks_str}.")
        context = [
            make_system_message(drink_recall_system_prompt),
            make_user_message(drink_recall_user_prompt.format(
                drinks=drinks_str,
                message=self.action.reply,
            ))
        ]
        recalls = self.gpt_bool_check(context, "recalls")
        recall_str = "recalled perfectly" if recalls else "forgot"
        self.reasoning.append(f"The agent {recall_str} what it was drinking.")
        self.score += int(recalls)

    def gpt_bool_check(self, context: LLMContext, key: str) -> bool:
        eval_json = self.ask_llm(context)
        try:
            eval_json = sanitize_and_parse_json(eval_json)
            return eval_json[key]
        except (ValueError, JSONDecodeError, KeyError) as exc:
            self.reasoning.append(f"Could not evaluate due to a JSON parsing error: {repr(exc)}")
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

This is what the menu looks like:

```text
{menu}
```

Extract the ordered items in JSON form, like this:
[
  "one item",
  "another ordered item",
  ...
]
""".strip()

items_in_menu_prompt = """
This is the menu at a restaurant:

```text
{menu}
```

And this is a diner's order:
{items}

For each of the items in the order, determine whether the item is contained in the menu or not.
Answer in JSON form, like this:
[
  ["one item", true or false],
  ["another item", true or false],
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

notice_mishap_prompt = """
You are an assistant in the customer experience department.
We have noticed that customers sometimes do not receive the meal that they have ordered.
Your goal is to discern whether the customer has noticed it or not.
Answer in JSON format, like this:
{"noticed": true or false}
""".strip()

drink_recall_system_prompt = """
You must evaluate whether the customer remembers what drinks it ordered before.
You will be given the original order, plus a sentence that the customer has just said to the waiter.

Answer in JSON form, like this:
{"recalls": true or false}
""".strip()

drink_recall_user_prompt = """
Original order: {drinks}
The customer says now: {message}
""".strip()
