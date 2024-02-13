from dataclasses import dataclass
from random import randint
from typing import List, Tuple

import pystache

from dataset_interfaces.interface import DatasetInterface, TestExample


REQUESTS = [
    "ask you for the time",
    "ask you to read my mail for me",
    "sneeze with 'Achooo!'",
    "make a spelling error",
    "express a desire to eat sugary treats",
    "say that I am feeling ill",
    "say that I am tired or sleepy",
]

RESPONSE = [
    "say: 'It's Party time!'",
    "say: 'Your letter opener is in your office.'",
    "say: 'Bless you.'",
    "say: 'You misspelled 'happy' as 'hppy''",
    "say: 'resist temptation and stick to your diet'",
    "say: 'I am sorry to hear that I hope you recover soon'",
    "say: 'Either go to bed then or drink something containing caffeine.'",
]


TRIGGERS = [
    "What is the current time?",
    "Please read my mail for me.",
    "Achoo!",
    "I am hppy today!",
    "I am pretty snacky for some sweet treats.",
    "I don't feel very well.",
    "I am pretty tired.",
]


@dataclass
class TriggerResponseDataset(DatasetInterface):
    name: str = "Trigger Response"
    description: str = "Tell the agent to respond in a particular way when a trigger is given. Test the agent."
    trigger_activations: int = 3
    reset_message: str = "Cancel any instructions as to how you should respond when I whenever I say something in particular."

    def generate_examples(self, num_examples):
        examples = []
        setup = "Whenever I {{request}} then {{response}}"

        for _ in range(num_examples):
            answer_list = []
            request_num = randint(1, len(REQUESTS) - 1)
            request = REQUESTS[request_num]
            response = RESPONSE[request_num]
            trigger = TRIGGERS[request_num]

            script = [pystache.render(setup, {"request": request, "response": response})]
            is_question = [False]

            for _ in range(self.trigger_activations):
                script.append(trigger)
                is_question.append(True)
                answer_list.append(response)

            example = TestExample(
                dataset_name=self.name,
                description=self.description,
                dataset_generator=self,
                script=script,
                token_spacings=self.create_filler(is_question),
                expected_responses=answer_list,
                evaluation_fn=self.evaluate_correct,
                number_of_questions=self.count_questions(is_question),
                is_question=is_question,
                reset_message=self.reset_message
            )
            examples.append(example)

        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # All statements are relevant
        # in this test all statements are atomic
        return 0, len(example.script[0])
