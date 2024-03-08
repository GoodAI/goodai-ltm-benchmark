from json import JSONDecodeError
from dataclasses import dataclass
from random import randint
from typing import List, Tuple
from utils.openai import make_user_message
from goodai.helpers.json_helper import sanitize_and_parse_json
from rouge_score.rouge_scorer import RougeScorer
import pystache

from dataset_interfaces.interface import DatasetInterface, TestExample


REQUESTS = [
    "ask you for the time",
    "ask you to read my mail for me",
    "sneeze with 'Achooo!'",
    "write 'hppy' instead of 'happy'",
    "express a desire to eat sugary treats",
    "say that I am feeling ill",
    "say that I am tired or sleepy",
]

RESPONSE = [
    "It's Party time!",
    "Your letter opener is in your office",
    "Bless you",
    "You misspelled 'happy' as 'hppy'",
    "resist temptation and stick to your diet",
    "I am sorry to hear that. I hope you recover soon",
    "Either go to bed then or drink something containing caffeine",
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

eval_prompt = """
Take a look at the following text:
"{message}"

Determine whether the sentence "{sentence}" is present or not in the text. If the sentence is present, extract the piece of text that features the targeted sentence. Answer in JSON form, like this:
{{"present": true or false, "sentence": "targeted sentence"}}
""".strip()


@dataclass
class TriggerResponseDataset(DatasetInterface):
    name: str = "Trigger Response"
    description: str = "Tell the agent to respond in a particular way when a trigger is given. Test the agent."
    trigger_activations: int = 3
    reset_message: str = "Cancel any instructions as to what sentence you should say whenever I do something in particular."

    def generate_examples(self, num_examples):
        examples = []
        setup = "Whenever I {{request}} then say: '{{response}}'"

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
                dataset_generator=self,
                script=script,
                expected_responses=answer_list,
                is_question=is_question,
            )
            examples.append(example)

        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        score = 0
        max_score = len(expected_answers)
        reasoning = list()
        for r, e in zip(responses, expected_answers):
            context = [make_user_message(eval_prompt.format(message=r, sentence=e))]
            eval_str = self.ask_llm(context)
            try:
                eval_json = sanitize_and_parse_json(eval_str)
                present = eval_json["present"]
                if present:
                    scorer = RougeScorer(["rougeL"], use_stemmer=True)
                    rouge_l = scorer.score(e, r)["rougeL"].fmeasure
                    present = rouge_l > 0.5
            except (ValueError, JSONDecodeError, KeyError) as exc:
                reasoning.append(f"Could not evaluate due to a JSON parsing error: {repr(exc)}")
                continue
            not_str = "" if present else "not "
            score += int(present)
            reasoning.append(f"'{e}' is {not_str}in the response.")
        return score, max_score, reasoning

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # All statements are relevant
        # in this test all statements are atomic
        return 0, len(example.script[0])
