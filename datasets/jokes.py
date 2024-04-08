import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

from dataset_interfaces.interface import DatasetInterface, TestExample, WaitCreator
from utils.timejump import create_time_jump


JOKES = [
    "My wife told me to stop impersonating a flamingo. I had to put my foot down.",
    "I went to buy some camo pants but couldn’t find any.",
    "I failed math so many times at school, I can’t even count.",
    "It takes a lot of balls to golf the way I do.",
    "Light travels faster than sound, which is the reason that some people appear bright before you hear them speak.",
    "A termite walks into the bar and asks, ‘Is the bar tender here?’",
    "Two fish are in a tank. One says, ‘How do you drive this thing?’",
    "Just burned 2,000 calories. That’s the last time I leave brownies in the oven while I nap.",
    "Atheism is a non-prophet organization",
]


@dataclass
class JokesDataset(DatasetInterface):
    name: str = "Jokes"
    description: str = "Tell the agent jokes, and fast forward through time, then ask for which joke was said around a particular moment."
    jokes_told: int = 3
    minutes_low: int = 30
    minutes_high: int = 240
    joke_prefix: bool = False

    def create_script_line(self, joke: str):
        return f"Joke: {joke}" if self.joke_prefix else joke

    def generate_examples(self, num_examples):
        examples = []
        for _ in range(num_examples):
            script = []
            selected_jokes = []
            is_question = []
            jokes = deepcopy(JOKES)
            waits = []

            filler_tokens = math.floor(self.memory_span * 0.75) // self.jokes_told
            for joke_made in range(self.jokes_told):
                if len(jokes) == 0:
                    logging.warning("Ran out of jokes")
                    break
                # Select joke
                joke = self.random.choice(jokes)
                jokes.remove(joke)

                # create the statement
                script.append(self.create_script_line(joke))
                selected_jokes.append(joke)
                is_question.append(False)
                time_jump = create_time_jump(self.minutes_low, self.minutes_high)
                waits.append(WaitCreator.create_wait(tokens=filler_tokens, time=time_jump))

            # Choose the joke we are going to look at
            answer = self.random.choice(selected_jokes)
            answer_list = [answer]

            # The question statement is generated dynamically
            is_question.append(True)
            waits.append({})

            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=answer_list,
                waits=waits,
                is_temporal=True,
                is_question=is_question,
            )

            examples.append(example)

        return examples

    def create_question(self, example: TestExample, statement_times, time_now):
        expected_response = example.expected_responses[0]
        expected_script_line = self.create_script_line(expected_response)
        answer_index = example.script.index(expected_script_line)
        said_time = statement_times[answer_index]
        delta = time_now - said_time
        minutes = delta.seconds // 60
        hours = minutes // 60

        timestamp = ""

        if minutes > 0:
            timestamp = f"{minutes % 60} minutes " + timestamp

        if hours > 0:
            timestamp = f"{hours} hours " + timestamp

        question = f"Which joke did I tell you about {timestamp} ago?"
        example.script.append(question)
        return question

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        if expected_answers[0] in responses[0]:
            max_score = 1
            score = 1
            reasons = ["The correct joke is recounted."]
            return score, max_score, reasons
        else:
            return self.evaluate_correct_gpt(questions, responses, expected_answers)


if __name__ == "__main__":
    gen = JokesDataset()
    joke_example = gen.generate_examples(2)
