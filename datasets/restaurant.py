import random
from typing import List, Tuple, Iterator

from datetime import datetime
from dataclasses import dataclass
from dataset_interfaces.interface import DynamicDataset, TestExample, DynamicExample, TestAction, SendMessageAction, WaitAction

from utils.openai import make_user_message
from goodai.helpers.json_helper import sanitize_and_parse_json


@dataclass
class RestaurantExample(DynamicExample):
    max_score: int = 1
    filler_tokens: int = 1000

    def action_iter(self) -> Iterator[TestAction]:

        # Setup, menu and ordering drinks.
        action = SendMessageAction(message=(
            "Now let's pretend to be at a restaurant. I will be the waiter and you will be the diner.\n"
            f"Good {day_moment_salutation()}. Welcome to our restaurant. Here is the menu for you to look over:\n\n"
            f"{menu}\n\nIn the meantime, what would you like to drink?"
        ))
        yield action
        drinks = self.extract_order_items(action.reply)
        drinks_str = enumerate_str(drinks)
        self.script[0] = action.message

        # Ordering food
        yield WaitAction(tokens=self.filler_tokens)
        action = SendMessageAction(message=f"Here is your {drinks_str}. What would you like to eat?")
        yield action
        order = self.extract_order_items(action.reply)
        order_str = enumerate_str(order).capitalize()
        yield SendMessageAction(
            message=f"Excellent choice! {order_str} coming right up."
        )

        # Some dish is unexpectedly unavailable
        yield WaitAction(tokens=self.filler_tokens)
        item = random.choice(order)
        order.remove(item)
        action = SendMessageAction(message=(
            f"I am very sorry, but I have been informed in the kitchen that the {item} is currently unavailable. "
            "Can I serve you something else instead?"
        ))
        yield action
        new_items = self.extract_order_items(action.reply)
        new_items_str = enumerate_str(new_items).capitalize()
        order.extend(new_items)
        yield SendMessageAction(message=f"{new_items_str} it is. Sorry again for the inconvenience.")

        # Deliver the meal after some time
        # TODO: deliver an extra thing or one less than expected
        # TODO: ask if it wants another drink and, most importantly, what the drink was.
        yield WaitAction(tokens=self.filler_tokens)
        order_str = enumerate_str(order)
        yield SendMessageAction(message=f"Here you are: {order_str}. Enjoy the meal.")

    def extract_order_items(self, message: str) -> list[str]:
        context = [make_user_message(extract_items_prompt.format(response=message))]
        items_json = self.ask_llm(context)
        return sanitize_and_parse_json(items_json)


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

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        return [RestaurantExample(
            dataset_generator=self,
            cost_callback=self._proxy_cost_callback
        ) for _ in range(num_examples)]

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


#####################
# Texts and Prompts #
#####################

menu = """
Appetizers:
1. Classic Caesar Salad
2. Crispy Calamari with Marinara Sauce
3. Bruschetta with Fresh Tomato, Basil, and Balsamic Glaze
4. Spinach and Artichoke Dip served with Tortilla Chips

Soups and Salads:
1. Soup of the Day (changes daily)
2. Garden Salad with Mixed Greens, Cucumbers, Tomatoes, and Balsamic Vinaigrette
3. French Onion Soup with Gruyere Cheese Crouton
4. Caprese Salad with Fresh Mozzarella, Tomatoes, Basil, and Balsamic Reduction

Entrees:
1. Grilled Salmon with Lemon Herb Butter, served with Roasted Vegetables and Rice Pilaf
2. Chicken Parmesan with Marinara Sauce and Melted Mozzarella, served with Spaghetti
3. Filet Mignon with Red Wine Demi-Glace, Garlic Mashed Potatoes, and Steamed Asparagus
4. Vegetarian Stir-Fry with Tofu, Mixed Vegetables, and Teriyaki Sauce over Steamed Rice

Pasta:
1. Spaghetti Carbonara with Pancetta, Egg, and Parmesan Cheese
2. Penne alla Vodka with Creamy Tomato Vodka Sauce
3. Linguine with Clams in White Wine Garlic Sauce
4. Vegetable Primavera with Seasonal Vegetables in a Light Tomato Sauce

Sandwiches:
1. Classic BLT with Crispy Bacon, Lettuce, Tomato, and Mayo on Toasted Sourdough
2. Grilled Chicken Club with Avocado, Bacon, Lettuce, Tomato, and Herb Mayo on a Brioche Bun
3. Turkey and Swiss Panini with Cranberry Aioli on Ciabatta Bread
4. Portobello Mushroom Burger with Roasted Red Peppers, Arugula, and Pesto Mayo on a Whole Wheat Bun

Vegan Options:
1. Vegan Lentil Soup with Seasonal Vegetables
2. Vegan Buddha Bowl with Quinoa, Roasted Chickpeas, Avocado, and Mixed Greens, drizzled with Tahini Dressing
3. Vegan Mushroom and Spinach Risotto with Arborio Rice and Truffle Oil
4. Vegan Beyond Burger with Lettuce, Tomato, Pickles, and Vegan Mayo on a Whole Wheat Bun, served with Sweet Potato Fries

Halal Options:
1. Halal Chicken Shawarma Plate with Grilled Chicken, Rice, Hummus, Salad, and Pita Bread
2. Halal Lamb Kebabs with Grilled Vegetables, Basmati Rice, and Tzatziki Sauce
3. Halal Beef Biryani with Fragrant Basmati Rice, Tender Beef, and Traditional Spices
4. Halal Falafel Wrap with Hummus, Lettuce, Tomato, Pickled Turnips, and Tahini Sauce in a Warm Pita

Desserts:
1. New York Style Cheesecake with Strawberry Compote
2. Warm Chocolate Lava Cake with Vanilla Ice Cream
3. Tiramisu with Espresso Soaked Ladyfingers and Mascarpone Cream
4. Fruit Tart with Seasonal Fresh Fruit and Pastry Cream

Beverages:
1. Soft Drinks (Coke, Diet Coke, Sprite, etc.)
2. Iced Tea (Sweetened or Unsweetened)
3. Lemonade
4. Freshly Brewed Coffee
5. Selection of Teas
6. Beer (Domestic and Imported)
7. Wine (Red, White, and Rosé by the Glass or Bottle)
""".strip()

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

recommendation_prompt = """
A waiter has been asked for a recommendation.
This is what the diner said:

```text
{question}
```

Please take a look at the menu and make a reasonable recommendation.

```menu
{menu}
```

Provide the recommendation as a single JSON string, like this:
"short name for one of the dishes in the menu"
""".strip()
