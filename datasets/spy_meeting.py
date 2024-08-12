import re
from dataclasses import dataclass
from typing import Tuple, List

from faker import Faker

from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.llm import make_user_message

PLACE_TEMPLATE = "{}: We will rendezvous {}."
TIME_TEMPLATE = "{}: The time we will meet is {}."
THING_TEMPLATE = "{}: Bring to the meeting {}."

CODED_INFO_PLACE = [
    ("where the land meets the sea", ["shore", "beach", "coastline"]),
    ("where the trains are supplied", ["railway", "yard", "depot"]),
    ("where the cars are made", ["automobile", "factory", "manufacturing"]),
    ("where the apples grow", ["orchard"]),
    ("where the sea cargo is stored", ["port", "warehouse", "seaport", "docks"]),
]

CODED_INFO_TIME = [
    ("when the sun starts its travel across the sky", ["dawn", "sunrise", "morning", "rise"]),
    ("when the blackbirds sing", ["dawn", "sunrise", "morning", "rise"]),
    ("when the sun is high", ["noon", "midday"]),
    ("when the sun leaves the sky", ["sunset", "dusk", "sets"]),
    ("when the moon is high", ["night", "midnight", "nighttime"]),
]

CODED_INFO_THING = [
    ("a way to get across a river", ["boat", "bridge", "raft", "kayak"]),
    ("a quiet way to open locked doors", ["pick", "picks", "key", "lockpicks", "lockpicking", "slim jim"]),
    ("a way to persuade the border guards to let us through", ["bribe", "credentials", "paperwork", "passport"]),
    ("a way to escape quickly over land", ["motorbike", "motorcycle", "car", "fast vehicle"]),
]


@dataclass
class SpyMeetingDataset(DatasetInterface):
    name: str = "Spy Meeting"
    description: str = "The agent is given three clandestine messages. Then is asked to recall all of them and decide where and when the meeting is taking place long with what they should bring."
    question: str = "Given the clandestine messages you have received, decypher their content and tell me when and where you think the meeting is going to happen and what you think you should bring. If you're unable to give a concrete time, place or item, give at least some examples."

    def generate_examples(self, num_examples):
        examples = []
        faker = Faker(["en_US", "en_IE"])
        Faker.seed(self.seed)

        for _ in range(num_examples):
            names = []
            expected_responses = []
            faker.unique.clear()
            for k in range(3):
                names.append(faker.unique.name())

            is_question = [False]
            script = [f"You will be given three messages from different people {names[0]}, {names[1]}, and {names[2]}."]
            topic_list = [(CODED_INFO_TIME, TIME_TEMPLATE), (CODED_INFO_THING, THING_TEMPLATE), (CODED_INFO_PLACE, PLACE_TEMPLATE)]

            for k in range(3):
                name = self.random.choice(names)
                names.remove(name)
                topic = self.random.choice(topic_list)
                topic_list.remove(topic)

                # A topic is a pair of coded statements and a template
                potential_messages, template = topic

                coded_message, potential_interpretations = self.random.choice(potential_messages)
                script.append(template.format(name, coded_message))

                is_question.append(False)
                expected_responses.append(potential_interpretations)

            script.append(self.question)
            is_question.append(True)

            examples.append(TestExample(
                dataset_generator=self,
                script=script,
                is_question=is_question,
                expected_responses=expected_responses,
            ))

        return examples

    def evaluate_correct(
            self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[float, int, List[str]]:

        reasoning = []
        response = responses[0]
        correct_score = 0
        incorrect_set = self.get_answers_for_others(expected_answers)

        for statement, potential_answers in zip(questions, expected_answers):
            found = False
            for pa in potential_answers:
                regex = re.compile(rf"\b{pa}\b")
                if len(regex.findall(response)) > 0:
                    found = True
                    break
            if not found:
                # Use an LLM to evaluate here
                prompt = """
You are evaluating whether the interpretation of a statement matches with the expected interpretation.
Here is the original statement:
{statement}

**********************************
Here is the interpretation:
{interpretation} 

**********************************
An expected interpretation should contain something that is synonymous to at least one of these items:
{expected}

**********************************
Respond with a single word. Either "Valid." for a valid interpretation, or "Invalid." for an invalid interpretation. 
A verbatim restatement of the original statement is always invalid.
""".strip()

                context = [make_user_message(prompt.format(statement=statement, interpretation=response, expected=potential_answers))]
                response_text = self.ask_llm(context=context, model="gpt-4-turbo", temperature=0.01)

                if response_text == "Valid.":
                    reasoning.append("Answer contains expected interpretation")
                    correct_score += 1
                else:
                    reasoning.append(f"{potential_answers} not found in answer.")
            else:
                reasoning.append("Answer contains expected keyword(s)")
                correct_score += 1

        # Check to see if the agent is confused
        confusion_score = 0
        for incorrect_term in incorrect_set:
            regex = re.compile(rf"\b{incorrect_term}\b")
            if len(regex.findall(response)) > 0:
                reasoning.append(f"Answer also contains `{incorrect_term}`, which indicates that the agent has recalled something incorrect.")
                confusion_score = 1

            # You only get marked down once for this
            if confusion_score == 1:
                break

        score = max((correct_score - confusion_score) / 3, 0.0)

        return score, 1, reasoning

    def get_answers_for_others(self, expected_answers):
        other_answers = set()
        for tup in CODED_INFO_TIME + CODED_INFO_THING + CODED_INFO_PLACE:
            for item in tup[1]:
                other_answers.add(item)

        for expected_list in expected_answers:
            for e in expected_list:
                other_answers.discard(e)

        return other_answers



if __name__ == '__main__':
    ds = SpyMeetingDataset()
    ex = ds.generate_examples(1)

    ex[0].script = []
    answer = """Based on the clandestine messages received, here is the deciphered content:1. **Frederick Ramos:** "Bring to the meeting a quiet way to open locked doors."2. **Mr. Terence Girvan:** "We will rendezvous where the sea cargo is stored."3. **Rachelle Cunny-Sheil:** "The time we will meet is when the sun is high."### Deciphered Meeting Details:**Time:**- "When the sun is high" typically refers to midday, around 12:00 PM.**Place:**- "Where the sea cargo is stored" suggests a location such as a port, dock, or warehouse near the sea where cargo is kept.**Item to Bring:**- A "quiet way to open locked doors" implies bringing a lock-picking set or a similar tool that can open locks discreetly.### Summary:- **Time:** Around 12:00 PM (midday)- **Place:** A port, dock, or warehouse where sea cargo is stored- **Item to Bring:** A lock-picking set or a similar discreet tool for opening locksIf you need further clarification or additional assistance, please let me know!"""