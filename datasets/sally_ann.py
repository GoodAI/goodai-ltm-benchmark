import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.constants import DATA_DIR


@dataclass
class SallyAnneDataset(DatasetInterface):
    name: str = "SallyAnne"
    description: str = "Give the agent a series of events. Then ask it a question about how the actors in those events would think."
    data_location: str = os.path.join(DATA_DIR, "tomi_data", "test.txt")

    def generate_examples(self, num_examples):
        examples = []
        for _ in range(num_examples):
            script = []
            is_question = []
            with open(self.data_location, "r") as f:
                f.seek(0, 2)
                b = f.tell()
                f.seek(int(b * random.random()))

                # Seek to start of next scenario
                line_num = ""
                current_line = ""
                while line_num != "1":
                    current_line = f.readline()
                    line_num = current_line.split(" ")[0]

                # Read out the scenario
                line_num = int(line_num)
                prev_line_num = 0
                while line_num > prev_line_num:
                    script.append(current_line)
                    is_question.append(False)
                    current_line = f.readline()
                    prev_line_num = line_num
                    line_num = int(current_line.split(" ")[0])

            is_question[-1] = True
            # Get the answer from the end of the scenario
            answer_list = [script[-1].split("?")[1].strip().split("\t")[0]]
            # Set the final statement to not have the answer
            script[-1] = script[-1].split("?")[0] + "?"

            # Remove numbers from the statements
            for idx, stmt in enumerate(script):
                script[idx] = " ".join(stmt.split(" ")[1:])

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
