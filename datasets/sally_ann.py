import re
from pathlib import Path
from json import JSONDecodeError
from dataclasses import dataclass
from typing import List, Tuple
from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.constants import DATA_DIR
from goodai.helpers.json_helper import sanitize_and_parse_json


patterns = dict(
    world_model=[
        r"^Where was the ([\w-]+) at the beginning\?$",
        r"^Where is the ([\w-]+) really\?$",
    ],
    theory_of_mind=[
        r"^Where will [\w-]+ look for the ([\w-]+)\?$",
        r"^Where does [\w-]+ think that [\w-]+ searches for the ([\w-]+)\?$",
    ],
)


def word_in_patterns(text: str, pattern_list: list[str]) -> str | None:
    for pattern in pattern_list:
        m = re.match(pattern, text)
        if m is not None:
            return m.group(1)


def is_plural(word: str) -> bool:
    """Works only for tomi_data/test.txt"""
    return word.endswith("s") and word not in {"asparagus", "dress", "pajamas"}


def fix_plural(sentence: str) -> str:
    if not sentence.startswith("The ") or " is in the " not in sentence:
        return sentence
    words = sentence.split(" ")
    if not is_plural(words[1]):
        return sentence
    words[2] = "are"
    return " ".join(words)


@dataclass
class SallyAnneDataset(DatasetInterface):
    name: str = "SallyAnne"
    description: str = "Give the agent a series of events. Then ask it a question about how the actors in those events would think."
    data_location: Path = DATA_DIR.joinpath("tomi_data", "test.txt")
    question_type: str = "any"  # world_model / theory_of_mind / any

    def __post_init__(self):
        self.samples = list()
        assert self.question_type in {"any", "world_model", "theory_of_mind"}
        question_types = ["world_model", "theory_of_mind"] if self.question_type == "any" else [self.question_type]
        self.question_patterns = [p for qt in question_types for p in patterns[qt]]
        self.extract_samples()

    def extract_samples(self):
        with open(self.data_location) as fd:
            sample_lines = list()
            for line in fd:
                # Remove index number
                line = line[line.find(" ") + 1:]
                # Add line to the script
                sample_lines.append(line)
                # Detect final line and wrap up sample
                if "\t" in line:
                    line, answer, _ = line.split("\t")
                    word = word_in_patterns(line, self.question_patterns)
                    if word is not None:
                        if is_plural(word):
                            for singular, plural in {"was": "were", "is": "was"}.items():
                                line = line.replace(singular, plural)
                        sample_lines[-1] = line
                        self.samples.append(dict(
                            lines=sample_lines,
                            answer=answer,
                        ))
                    sample_lines = list()

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        examples = list()
        samples = self.samples.copy()
        self.random.shuffle(samples)
        for i, sample in enumerate(samples[:num_examples]):
            script = [
                "They are broadcasting a program on TV. I will keep you updated on what happens, and at the end, I will "
                "ask you a question about what happened on the show. Okay?"
            ]
            script.extend(f"(On TV) {fix_plural(line)}" for line in sample["lines"][:-1])
            script.append(
                f"The TV program has ended for today. {sample['lines'][-1]}\n"
                'Provide your answer in JSON form with a single word as answer, like this: {"answer": "word"}\n'
                "Be as specific as possible."
            )
            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=[sample["answer"]],
                is_question=[False] * (len(script) - 1) + [True],
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

        if isinstance(answer_dict, dict) and answer_dict.get("answer", "").lower() == expected_answers[0]:
            score = 1
            reasoning = f"The agent answered with {repr(answer_dict['answer'])}, which is the right answer."
        else:
            reasoning = f"Invalid answer: {answer_dict}"
        return score, max_score, [reasoning]

