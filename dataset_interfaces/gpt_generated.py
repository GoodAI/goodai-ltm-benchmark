import json
from abc import ABC
from dataclasses import dataclass

import pystache

from dataset_interfaces.interface import DatasetInterface, TestExample, CallBackTestExample
from utils.json_helper import sanitize_and_parse_json
from utils.openai import ask_llm

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
    generation_file: str = ""
    temperature: float = 1.0
    uses_callback: bool = False

    def generate_examples(self, num_examples):
        examples = []
        with open(self.generation_file, "r") as f:
            prompt_data = json.loads("".join(f.readlines()))

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
            for _ in range(10):
                try:
                    result = ask_llm(context, temperature=self.temperature)
                    generated = sanitize_and_parse_json(result)
                    correct = True
                    break
                except:
                    pass
            if not correct:
                raise ValueError(
                    "GPT powered generation failed after 10 attempts! You can choose to rerun the generation."
                )

            script.append("\n".join(generated["content"]))
            is_question.append(False)

            for q, a in zip(generated["question"], generated["answer"]):
                is_question.append(True)
                script.append(q)
                expected_responses.append(a)

            example_class = CallBackTestExample if self.uses_callback else TestExample

            example = example_class(
                dataset_name=self.name,
                description=self.description,
                dataset_generator=self,
                script=script,
                expected_responses=expected_responses,
                evaluation_fn=self.evaluate_correct,
                token_spacings=self.create_filler(is_question),
                can_be_interleaved=True,
                number_of_questions=self.count_questions(is_question),
                is_question=is_question,
                uses_callback=self.uses_callback,
            )

            examples.append(example)

        return examples
