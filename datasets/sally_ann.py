import os.path
from json import JSONDecodeError
from dataclasses import dataclass
from typing import List, Tuple
from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.constants import DATA_DIR
from goodai.helpers.json_helper import sanitize_and_parse_json


def next_line(file_descriptor) -> str:
    pos = file_descriptor.tell()
    line = file_descriptor.readline()
    if file_descriptor.tell() == pos:
        file_descriptor.seek(0)
        line = file_descriptor.readline()
    return line


@dataclass
class SallyAnneDataset(DatasetInterface):
    name: str = "SallyAnne"
    description: str = "Give the agent a series of events. Then ask it a question about how the actors in those events would think."
    data_location: str = os.path.join(DATA_DIR, "tomi_data", "test.txt")

    def generate_examples(self, num_examples):
        examples = []
        for _ in range(num_examples):
            script = [
                "They are broadcasting a program on TV. I will keep you updated on what happens, and at the end, I will "
                "ask you a question about what happened on the show. Okay?"
            ]
            is_question = [False]
            with open(self.data_location, "r") as f:
                num_chars = f.seek(0, 2)
                f.seek(self.random.randint(0, num_chars - 1))
                next_line(f)  # Read until the new line starts

                # Seek to start of next scenario
                while True:
                    line = next_line(f)
                    if line[:2] == "1 ":
                        break

                # Read out the scenario
                while True:
                    line = line[line.find(" ") + 1:]  # Remove numbers
                    script.append(line)
                    is_question.append(False)
                    line = next_line(f)
                    if line[:2] == "1 ":
                        break

            is_question[-1] = True
            script[-1], answer, _ = script[-1].split("\t")

            # Format script
            for idx, stmt in enumerate(script[1:-1]):
                script[idx + 1] = f"(On TV) {stmt}"
            script[-1] = (
                f"The TV program has ended for today. {script[-1]}\n"
                'Provide your answer in JSON form with a single word as answer, like this: {"answer": "word"}'
            )

            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=[answer],
                is_question=is_question,

            )
            examples.append(example)

        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:

        score = 0
        max_score = 1

        try:
            answer_dict = sanitize_and_parse_json(responses[0])
        except (ValueError, JSONDecodeError) as exc:
            reasoning = f"Invalid answer. {exc}"
            return score, max_score, [reasoning]

        if isinstance(answer_dict, dict) and answer_dict.get("answer", "") == expected_answers[0]:
            score = 1
            reasoning = f"The agent answered with {repr(answer_dict['answer'])}, which is the right answer."
        else:
            reasoning = f"Invalid answer: {answer_dict}"
        return score, max_score, [reasoning]

