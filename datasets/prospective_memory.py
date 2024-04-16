import string
from dataclasses import dataclass
from typing import List, Tuple, Any, Iterator
from utils.ui import ordinal, colour_print

from dataset_interfaces.interface import TestExample, DatasetInterface, CallBackTestExample

QUOTES = [
    ("Love your Enemies, for they tell you your Faults.", "Benjamin Franklin"),
    ("The greatest glory in living lies not in never falling, but in rising every time we fall.", "Nelson Mandela"),
    ("The future belongs to those who believe in the beauty of their dreams.", "Elanor Roosevelt"),
    ("Do one thing every day that scares you.", "Eleanor Roosevelt"),
    ("Well done is better than well said.", "Benjamin Franklin"),
    ("The best and most beautiful things in the world cannot be seen or even touched - they must be felt with the heart.",
    "Helen Keller"),
    ("It is during our darkest moments that we must focus to see the light.", "Aristotle"),
    ("Do not go where the path may lead, go instead where there is no path and leave a trail.", "Ralph Waldo Emerson"),
    ("Be yourself; everyone else is already taken.", "Oscar Wilde")
]



def cites_quote(quote: str, message: str) -> bool:
    table = str.maketrans("", "", string.punctuation)
    quote = quote.lower().translate(table)
    message = message.lower().translate(table)
    return quote in message


@dataclass
class ProspectiveMemoryDataset(DatasetInterface):
    name: str = "Prospective Memory"
    description: str = "Give the agent a quote, then ask it to append to the nth reply"
    reset_message: str = "Forget my instruction to append a quote to one of your replies."

    def request_num_generator(self) -> Iterator[int]:
        indices = list(range(len(QUOTES)))
        warned = False
        while True:
            self.random.shuffle(indices)
            for i in indices:
                yield i
            if warned:
                continue
            warned = True
            colour_print("red",
                         "WARNING: The number of Quotes has been exceeded. There will inevitably "
                         "be some repetitions.")

    def generate_examples(self, num_examples) -> List[TestExample]:
        examples = []

        for _, request_num in zip(range(num_examples), self.request_num_generator()):
            script = []
            # Pick a quote
            quote, author = QUOTES[request_num]
            target = self.random.randint(2, 8)

            n = ordinal(target)

            script.append(f"'{quote}' - {author}")
            script.append(f'After responding to the following unrelated user prompts, please append the quote from {quote}, to your {n} response. Count your response to this message as the first response.')

            is_question = [False, True]
            expected_responses = [(quote, author, target)]

            examples.append(CallBackTestExample(
                dataset_generator=self,
                script=script,
                expected_responses=expected_responses,
                uses_callback=True,
                is_question=is_question,
                number_of_questions=1,
            ))

        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str], List[str]]:
        raise NotImplementedError("Prospective memory checking is not handled by this method, use the callback instead")

    def get_log(self, scheduler, example: TestExample) -> list[str]:
        first_message = example.script[0]
        start_idx = 0
        all_messages = scheduler.master_log.messages()
        for idx, msg in enumerate(all_messages):
            if first_message in msg:
                start_idx = idx
                break

        return all_messages[start_idx:]

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

        quote, author, nth = example.expected_responses[0]
        agent_responses = task_log[statement_idx + 1:][::2]
        response_w_quote_idx = nth - 1

        max_score = 1

        # If the quote hasn't come up yet
        if response_w_quote_idx >= len(agent_responses):
            score = 1
            reason = "Not yet seen"
            deregister_callback = False
            example.finished = False

            return score, max_score, [reason], deregister_callback

        # Get the quote
        quote = example.expected_responses[0][0]

        # This statement should have the quote attached
        target_stmt_in_log = agent_responses[response_w_quote_idx]
        if cites_quote(quote, target_stmt_in_log):
            score = 1
            reason = "The quote is recited in the correct place."
            deregister_callback = True
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

