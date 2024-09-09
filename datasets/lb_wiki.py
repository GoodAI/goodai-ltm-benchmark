import json
from dataclasses import dataclass
from typing import List, Any, Tuple

from goodai.helpers.json_helper import sanitize_and_parse_json

from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.llm import make_user_message


@dataclass
class LongBenchWikiQADataset(DatasetInterface):

    name: str = "WikiQA"
    description: str = ""
    characters_per_chunk = 1500

    def generate_examples(self, num_examples):

        with open("data/2wikimqa_e.jsonl", "r", encoding="utf-8", errors="ignore") as fd:
            data_lines = self.random.sample(fd.readlines(), k=num_examples)

        examples = []

        for l in data_lines:
            json_struct = json.loads(l)

            is_question=[]
            script = []
            idx1 = 0
            idx2 = 0
            script_line = json_struct["context"]
            while idx2 < len(script_line):
                idx2 += self.characters_per_chunk
                while idx2 < len(script_line) and script_line[idx2] not in [",", " ", ".", "?", "!"]:
                    idx2 += 1

                script.append(script_line[idx1:idx2])
                is_question.append(False)
                idx1=idx2

            question = f"Answer the question based on the previously given information. Only give me the answer and do not output any other words. Question: {json_struct['input']}"
            script.append(question)
            is_question.append(True)


            examples.append(TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=json_struct["answers"],
                is_question=is_question,
                script_is_filler=True,
            ))

        return examples

    def evaluate_correct(self, questions: List[str], responses: List[str], expected_answers: List[Any]) -> Tuple[int, int, List[str]]:

        # We ask a relatively open fact based question now we need to evaluate the answer using a GPT model
        answer_prompt = """Judge whether a given answer matches the expected answer. Focus on the core facts and ignore any extra detail in in the given answer

EXPECTED ANSWER:
{expected_answer}

***********************
GIVEN ANSWER
{given_answer}        

Reply JSON like this:
{{
    "correct": bool // Whether the facts of the given answer match those in the expected answer.
}}        
"""
        context = [make_user_message(answer_prompt.format(expected_answer=expected_answers[0], given_answer=responses[0]))]
        judgement_str = self.ask_llm(context, model="gpt-4o-mini")
        judgement_json = sanitize_and_parse_json(judgement_str)

        if judgement_json["correct"]:
            reason = "The answer is judged to be correct,"
            score = 1
        else:
            reason = "The answer is judged to be incorrect."
            score = 0

        return score, 1, [reason]

