from dataclasses import dataclass
from random import choice
from typing import Tuple, List

from faker import Faker

from dataset_interfaces.interface import DatasetInterface, TestExample, WaitCreator


PLACE_TEMPLATE = "{}: We will rendezvous {}."
TIME_TEMPLATE = "{}: The time we will meet is {}."
AMOUNT_TEMPLATE = "{}: Bring to the meeting {}."

CODED_INFO_PLACE = [
    ("where the land meets the sea", ["shore", "beach", "coastline"]),
    ("where the occupiers supply their trains", ["railway depot", "train yard"]),
    ("where the cars are made", ["automobile factory", "car factory"]),
    ("where the apples grow", ["orchard"]),
    ("at the most geometric part of the city", ["city square"])
]

CODED_INFO_TIME = [
    ("when the sun starts its travel across the sky", ["sunrise", "dawn"]),
    ("when the blackbirds sing", ["dawn", "sunrise"]),
    ("when the sun is high", ["noon"]),
    ("when the sun leaves the sky", ["sunset", "dusk"]),
    ("when the moon is high", ["night"]),
]

CODED_INFO_AMOUNT = [
    ("a way to get across a river", ["boat", "bridge"]),
    ("a quiet way to open locked doors", ["lockpicks", "master key"]),
    ("a way to persuade the border guards to let us through", ["bribe", "credentials"]),
    ("a fast land escape vehicle", ["motorbike", "fast car"]),
]


@dataclass
class SpyMeetingDataset(DatasetInterface):
    name: str = "Spy Meeting"
    description: str = "The agent is given three clandestine messages, "
    question: str = "Given the clandestine messages you have recieved, tell me when and where a meeting is going to happen and what you should bring."

    def generate_examples(self, num_examples):
        examples = []
        faker = Faker(["en_US", "en_IE"])

        for _ in range(num_examples):
            names = []
            expected_responses = []
            messages = []
            faker.unique.clear()
            for k in range(3):
                names.append(faker.unique.name())

            waits = [WaitCreator.create_wait(percentage_finished=10)]
            is_question = [False]
            script = [f"You will be given three messages from different people {names[0]}, {names[1]}, and {names[2]}."]
            topic_list = [(CODED_INFO_TIME, TIME_TEMPLATE), (CODED_INFO_AMOUNT, AMOUNT_TEMPLATE), (CODED_INFO_PLACE, PLACE_TEMPLATE)]

            for k in range(3):
                name = choice(names)
                names.remove(name)
                topic = choice(topic_list)
                topic_list.remove(topic)

                # A topic is a pair of coded statements and a template
                potential_messages, template = topic

                coded_message, potential_interpretations = choice(potential_messages)
                message = template.format(name, coded_message)

                messages.append(message)
                script.append(message)

                is_question.append(False)
                expected_responses.append(potential_interpretations)

                if k == 2:
                    waits.append(WaitCreator.create_wait(percentage_finished=90))
                else:
                    waits.append(WaitCreator.create_wait(percentage_finished=k+2 * 10))

            script.append(self.question)
            is_question.append(True)
            waits.append(waits)

            examples.append(TestExample(
                dataset_generator=self,
                script=script,
                waits=waits,
                is_question=is_question,
                expected_responses=expected_responses
            ))

        return examples

    def evaluate_correct(
            self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:

        reasoning = []
        response = responses[0]
        correct = 1
        for potential_answers in expected_answers:
            found = False
            for pa in potential_answers:
                if pa in response:
                    found = True
                    break
            if not found:
                correct = 0
                reasoning.append(f"{potential_answers} not found in answer.")

        return correct, 1, reasoning

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # Second statement in the script, character 0
        return 1, 0
