from dataclasses import dataclass
from typing import Tuple, List

from faker import Faker

from dataset_interfaces.interface import DatasetInterface, TestExample, WaitCreator
# from utils.openai import ask_llm

PLACE_TEMPLATE = "{}: We will rendezvous {}."
TIME_TEMPLATE = "{}: The time we will meet is {}."
THING_TEMPLATE = "{}: Bring to the meeting {}."

CODED_INFO_PLACE = [
    ("where the land meets the sea", ["shore", "beach", "coastline"]),
    ("where the trains are supplied", ["railway", "yard", "depot"]),
    ("where the cars are made", ["automobile", "factory", "manufacturing"]),
    ("where the apples grow", ["orchard"]),
    ("where the sea cargo is stored", ["port", "warehouse"]),
]

CODED_INFO_TIME = [
    ("when the sun starts its travel across the sky", ["sunrise", "dawn", "rise"]),
    ("when the blackbirds sing", ["dawn", "sunrise", "morning", "rise"]),
    ("when the sun is high", ["noon", "midday"]),
    ("when the sun leaves the sky", ["sunset", "dusk", "set"]),
    ("when the moon is high", ["night", "midnight"]),
]

CODED_INFO_THING = [
    ("a way to get across a river", ["boat", "bridge", "raft", "kayak"]),
    ("a quiet way to open locked doors", ["pick", "key"]),
    ("a way to persuade the border guards to let us through", ["bribe", "credentials", "paperwork", "passport"]),
    ("a way to escape quickly over land", ["motorbike", "motorcycle", "car", "fast vehicle"]),
]


@dataclass
class SpyMeetingDataset(DatasetInterface):
    name: str = "Spy Meeting"
    description: str = "The agent is given three clandestine messages. Then is asked to recall all of them and decide where and when the meeting is taking place long with what they should bring."
    question: str = "Given the clandestine messages you have received, tell me as specifically as you can, when and where a meeting is going to happen and what you should bring."

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

            waits = [WaitCreator.create_wait(percentage_finished=10)]
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

                if k == 2:
                    waits.append(WaitCreator.create_wait(percentage_finished=90))
                else:
                    waits.append(WaitCreator.create_wait(percentage_finished=(k+2) * 10))

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
            else:
                reasoning.append("Answer contains expected keyword(s)")

        return correct, 1, reasoning

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # Second statement in the script, character 0
        return 1, 0


