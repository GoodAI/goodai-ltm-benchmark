from json import JSONDecodeError
from typing import Iterator
from collections import OrderedDict
from dataclasses import dataclass, field
from dataset_interfaces.interface import DynamicDataset, DynamicExample, TestAction, SendMessageAction
from utils.llm import make_system_message, make_user_message, LLMContext, GPT_4_TURBO_BEST, GPT_CHEAPEST
from goodai.helpers.json_helper import sanitize_and_parse_json


class RestaurantOrderFailed(Exception):
    pass


@dataclass
class RestaurantExample(DynamicExample):
    dataset_generator: "RestaurantDataset" = None
    max_score: int = 5  # 1 point max. per sub-challenge
    messages: list[str] = field(default_factory=list)

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
            "When I talk to you as the waiter (e.g. 'Waiter: what will it be sir?'), then you will reply as if you were the "
            "customer at a restaurant. Give straight answers to the questions and avoid going off script. Understood?"
        )
        yield self.wait(percentage_finished=20)

        # Give the menu and ask for the drink
        self.messages.append("Waiter: In the meantime, what would you like to drink?")
        yield self.say(
            "Good day. Welcome to our restaurant. Here is the menu for you to look over:\n\n"
            f"{self.dataset_generator.menu}\n\nIn the meantime, what would you like to drink?",
        )
        self.expected_responses.append("The agent follows the role of a customer at a restaurant and orders a drink.")
        self.check_role_following()
        self.messages.append(f"Diner: {self.action.reply}")
        drinks = self.extract_drink_order(self.action.reply)
        drinks_str = enumerate_str(drinks)
        self.reasoning.append(f"The agent answered as the customer and ordered {drinks_str}.")
        self.score += 1
        yield self.wait(percentage_finished=40)

        # Ordering food
        self.messages.append(f"Waiter: Here is your {drinks_str}. What would you like to eat?")
        yield self.say(f"Here is your {drinks_str}. What would you like to eat?")
        self.expected_responses.append("The agent orders at least one dish from the menu.")
        self.messages.append(f"Diner: {self.action.reply}")
        order = self.extract_order_items(self.action.reply)

        order_str = enumerate_str(order)
        self.reasoning.append(f"The agent ordered {order_str}.")
        self.score += 1
        self.messages.append(f"Waiter: Excellent choice! {order_str} coming right up.")
        yield self.say(f"Excellent choice! {order_str} coming right up.", question=False)
        yield self.wait(percentage_finished=60)

        # Some dish is unexpectedly unavailable -> order another thing
        old_item = self.random.choice(order)
        self.messages.append(f"Waiter: I am very sorry, but I have been informed in the kitchen that the {old_item} is currently "
            "unavailable. Can I serve you something else instead?")
        yield self.say(
            f"I am very sorry, but I have been informed in the kitchen that the {old_item} is currently "
            "unavailable. Can I serve you something else instead?"
        )
        self.expected_responses.append("The agent orders an alternative meal from the menu.")
        new_items = self.extract_order_items(self.action.reply)
        new_items_str = enumerate_str(new_items)
        repeated_items = [item for item in new_items if item in order]
        if len(repeated_items) > 0:
            self.reasoning.append(f"The agent orders some things again: {repeated_items}")
            return
        self.reasoning.append("The agent orders a different meal.")
        self.score += 1

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

    def extract_drink_order(self, message: str):
        conversation = "".join(self.messages)
        context = [make_user_message(extract_items_prompt.format(conversation=conversation, menu=self.dataset_generator.menu))]
        response_json = self.ask_llm(context, model=GPT_4_TURBO_BEST)

        try:
            response = sanitize_and_parse_json(response_json)
        except (ValueError, JSONDecodeError):
            self.reasoning.append("Could not extract ordered items due to a JSON parse error.")
            raise RestaurantOrderFailed

        items = list()
        for item_dict in response["order"]:
            if item_dict["is_drink"]:
                items.append(item_dict["item"])
                continue

        if not response["has_ordered_something"] or len(items) == 0:
            self.reasoning.append("The agent did not order anything.")
            raise RestaurantOrderFailed

        return items

    def extract_order_items(self, message: str) -> list[str]:
        conversation = "".join(self.messages)
        context = [make_user_message(extract_items_prompt.format(conversation=conversation, menu=self.dataset_generator.menu))]
        response_json = self.ask_llm(context, model=GPT_4_TURBO_BEST)

        try:
            response = sanitize_and_parse_json(response_json)
        except (ValueError, JSONDecodeError):
            self.reasoning.append("Could not extract ordered items due to a JSON parse error.")
            raise RestaurantOrderFailed

        items = list()
        for item_dict in response["order"]:
            if item_dict["is_drink"]:
                continue
            if item_dict["off_menu"]:
                self.reasoning.append(f"{item_dict['item']} is not in the menu.")
                raise RestaurantOrderFailed
            menu_nr = item_dict["menu_nr"]
            if isinstance(menu_nr, str):
                menu_nr = int(menu_nr.strip())
            if not (1 <= menu_nr <= len(self.dataset_generator.menu_items)):
                self.reasoning.append(f"{item_dict['item']} is not in the menu.")
                raise RestaurantOrderFailed
            items.append(self.dataset_generator.menu_items[menu_nr - 1])

        if not response["has_ordered_something"] or len(items) == 0:
            self.reasoning.append("The agent did not order anything.")
            raise RestaurantOrderFailed

        return items

    def find_alternative_dish(self, item: str, old_item: str) -> tuple[str, str] | None:
        for section_content in self.dataset_generator.menu_dict.values():
            for section_item in section_content:
                if item in section_item:
                    choices = [c for c in section_content if c not in [section_item, old_item]]
                    return section_item, self.random.choice(choices)

    def alter_order(self, order: list[str], old_item: str) -> tuple[str, str, list[str]]:
        sh_order = order.copy()
        self.random.shuffle(sh_order)
        for item in sh_order:  # Sometimes there is a drink in the order, which is not in the menu.
            alternative = self.find_alternative_dish(item, old_item)
            if alternative is not None:
                orig_dish, alt_dish = alternative
                altered_order = [item for item in order]
                i = altered_order.index(item)
                altered_order[i] = alt_dish
                return orig_dish, alt_dish, altered_order
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
        follows_role = self.gpt_bool_check(context, "follows_role", model=GPT_4_TURBO_BEST)
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

    def gpt_bool_check(self, context: LLMContext, key: str, model: str = GPT_CHEAPEST, **llm_kwargs) -> bool:
        eval_json = self.ask_llm(context, model, **llm_kwargs)
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
        self.menu_items = [item for content in self.menu_dict.values() for item in content]
        self.menu: str = "\n\n".join(
            "\n".join([f"{section}:"] + [f"{self.menu_items.index(item) + 1}. {item}" for item in content])
            for section, content in self.menu_dict.items()
        )


def enumerate_str(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return " and ".join([", ".join(items[:-1]), items[-1]])


###########
# Prompts #
###########

extract_items_prompt = """
This is a conversation between diner and a waiter:

```text
{conversation}
```

This is what the menu looks like:

```text
{menu}
```

Read carefully the diner's final response only, note what has been newly ordered and not mentioned before, and fill in this JSON form:
{{
  "has_ordered_something": true or false,
  "order": [
    {{
      "is_drink": true or false,
      "item": "ordered item",
      "menu_nr": (int or null) item number in menu,
      "menu_ref": "verbatim name as it appears in the menu",
      "off_menu": true if the item is not in the menu or alters the dish
    }},
    ...
  ]
}}

Shortened forms are not considered alterations of menu dishes. For example: kebabs with vegetables and rice; chicken parmesan with spaguetti.

Forms that introduce new elements or mix dishes are considered alterations. For example: grilled chicken club with crispy bacon.
""".strip()

role_eval_prompt = """
You are an evaluator in an acting session.
You evaluate how well participants perform under a previously agreed setting.

The current setting is: the participant is a human customer at a restaurant; the waiter has just given them the menu and asked them what they want to drink. The customer is expected to go along the situation and order something to drink.

You must say whether the participant followed the assigned role or not. Respond in JSON form, like this:
{"follows_role": true or false}
""".strip()

notice_mishap_prompt = """
You are an assistant in the customer experience department.
We have noticed that customers sometimes do not receive the meal that they have ordered.
Your goal is to discern whether the customer has noticed it or not.
Beware that some clients will notice it but won't actually complain.
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
