import re
import string
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any
from utils.ui import ordinal

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
    generation_model: str = "gpt-4-0125-preview"

    def generate_examples(self, num_examples) -> List[TestExample]:
        num_pattern = r"\d+(?:th|st|nd|rd)"
        examples = super().generate_examples(num_examples)
        for example in examples:
            n = ordinal(self.random.randint(2, 9))
            example.script[1] = re.sub(num_pattern, n, example.script[1])
            example.script[1] += " Count your response to this message as the first response."
            example.expected_responses[0] = re.sub(num_pattern, n, example.expected_responses[0])
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

        agent_responses = task_log[statement_idx + 1:][::2]
        response_w_quote_idx = int(re.findall(r"\d+", question)[0]) - 1

        max_score = 1

        # If the quote hasn't come up yet
        if response_w_quote_idx >= len(agent_responses):
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
        target_stmt_in_log = agent_responses[response_w_quote_idx]
        if cites_quote(quote, target_stmt_in_log):
            score = 1
            reason = "The quote is recited in the correct place."
            deregister_callback = False
        else:
            score = 0
            reason = "The agent did not recite the quote in the correct place."
            deregister_callback = True
        example.finished = True

        # All other statements should not have the quote
        for i, stmt in enumerate(agent_responses):
            if i == response_w_quote_idx:
                continue
            if cites_quote(quote, stmt):
                score = 0
                reason = "The quote is recited somewhere other or additionally to the correct place."
                deregister_callback = True
                break

        return score, max_score, [reason], deregister_callback


if __name__ == "__main__":
    s = ProspectiveMemoryDataset()
    examples = s.generate_examples(4)
    a = 1
