import re
import random
import string
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any


from dataset_interfaces.gpt_generated import GPTGenerated

from dataset_interfaces.interface import TestExample
from utils.constants import DATA_DIR


def cites_quote(quote: str, message: str) -> bool:
    table = str.maketrans("", "", string.punctuation)
    quote = quote.lower().translate(table)
    message = message.lower().translate(table)
    return quote in message


@dataclass
class ProspectiveMemoryDataset(GPTGenerated):
    name: str = "Prospective Memory"
    description: str = "Give the agent a fictitious quote, then ask it to append to the nth reply"
    generation_file: Path = DATA_DIR.joinpath("gpt_generation_prompts/2-1_prospective_memory_test.json")
    temperature: float = 0.5
    uses_callback: bool = True
    generation_model: str = "gpt-4-1106-preview"

    def generate_examples(self, num_examples) -> List[TestExample]:
        rnd = random.Random(self.seed)
        examples = super().generate_examples(num_examples)
        for example in examples:
            n = str(rnd.randint(1, 10))
            example.script[1] = re.sub(r"\d+", n, example.script[1])
            example.expected_responses[0] = re.sub(r"\d+", n, example.expected_responses[0])
        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str]]:
        raise NotImplementedError("Prospective memory checking is not handled by this method, use the callback instead")

    def continual_evaluation_callback(
        self, scheduler, example: TestExample, task_log: List[str]
    ) -> Tuple[int, int, List[str], bool]:
        # Find where we asked in the master log and how many steps we need to look into the future
        question = example.script[-1]
        statement_idx = None
        for idx, stmt in enumerate(task_log):
            if question in stmt:
                statement_idx = idx
                break

        # Log to actually check
        log_lookahead = task_log[statement_idx:]
        # steps to look
        steps_to_look = int(re.findall(r"\d+", question)[0])
        # We need to adjust this, because prompts are [User -> Agent] and +1 for the question statement and response
        steps_to_look = (steps_to_look + 1) * 2

        max_score = 1

        # If the quote hasn't come up yet
        if steps_to_look > len(log_lookahead):
            score = 1
            reason = "Not yet seen"
            deregister_callback = False
            example.finished = False

            return score, max_score, [reason], deregister_callback

        # Get the quote
        quote = None
        for quote_chars in ["''", '""', "``", "<>"]:
            open_char, close_char = quote_chars
            pattern = re.compile(f"{re.escape(open_char)}(.*?){re.escape(close_char)}")
            matches = pattern.findall(example.script[0])
            if len(matches) > 0:
                quote = matches[0].lower()
                break
        assert quote is not None, "Quote cannot be found"

        # This statement should have the quote attached
        target_stmt_in_log = log_lookahead[steps_to_look - 1]
        if cites_quote(quote, target_stmt_in_log):
            score = 1
            reason = "The quote is recited in the correct place."
            deregister_callback = False
            example.finished = True
        else:
            score = 0
            reason = "The agent did not recite the quote in the correct place."
            deregister_callback = True
            example.finished = True

        # All other statements should not have the quote
        for stmt in log_lookahead:
            if stmt.lower() == target_stmt_in_log.lower():
                continue

            if cites_quote(quote, stmt):
                score = 0
                reason = "The quote is recited somewhere other or additionally to the correct place."
                deregister_callback = True
                example.finished = True
                break

        return score, max_score, [reason], deregister_callback


if __name__ == "__main__":
    s = ProspectiveMemoryDataset()
    examples = s.generate_examples(4)
    a = 1
