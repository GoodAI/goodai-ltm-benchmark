from abc import ABC
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any, Tuple

import pystache
from goodai.helpers.json_helper import sanitize_and_parse_json

from dataset_interfaces.interface import DatasetInterface, TestExample, CallBackTestExample
from utils.llm import ask_llm, GPT_CHEAPEST

PROMPT = """Generate data and questions based on the structure and instructions below.
- For content: {{content}} 
- For question: {{question}}
- For answer: {{answer}}

Structure your response as such:
{
    "content": [""],
    "question": [""],
    "answer": [""]
}"""


@dataclass
class GPTGenerated(DatasetInterface, ABC):
    generation_file: str | Path = None
    temperature: float = 1.0
    generation_model: str = GPT_CHEAPEST
    max_attempts: int = 10

    def __post_init__(self):
        if self.generation_file is None:
            raise ValueError("GPTGenerated datasets require a file path to read from.")

    def generate_examples(self, num_examples) -> List[TestExample]:
        examples = []
        prompt_data = self.load_json(self.generation_file)

        for _ in range(num_examples):
            script = []
            is_question = []
            expected_responses = []
            # Generate a set of examples using GPT
            generation_prompt = pystache.render(PROMPT, prompt_data)
            context = [
                {"role": "system", "content": "You are a helpful creative agent"},
                {"role": "user", "content": generation_prompt},
            ]
            correct = False
            for _ in range(self.max_attempts):
                try:
                    result = ask_llm(context, temperature=self.temperature, model=self.generation_model)
                    generated = sanitize_and_parse_json(result)
                    correct = True
                    break
                except:
                    pass
            if not correct:
                raise ValueError(
                    f"GPT powered generation failed after {self.max_attempts} attempts! You can choose to rerun the generation."
                )

            script.append("\n".join(generated["content"]))
            is_question.append(False)

            for q, a in zip(generated["question"], generated["answer"]):
                is_question.append(True)
                script.append(q)
                expected_responses.append(a)

            example_class: type[TestExample] = CallBackTestExample if self.uses_callback else TestExample

            example = example_class(
                dataset_generator=self,
                script=script,
                expected_responses=expected_responses,
                is_question=is_question,
                uses_callback=self.uses_callback,
            )

            examples.append(example)

        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)
