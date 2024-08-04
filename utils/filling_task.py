import json
from random import Random
from typing import Callable

import tiktoken
from model_interfaces.interface import ChatSession
from utils.constants import DATA_DIR


TRIVIA_CACHE = None


def get_trivia() -> list[dict[str, str]]:
    global TRIVIA_CACHE
    if TRIVIA_CACHE is None:
        with (open(DATA_DIR.joinpath("trivia/trivia.json"), "r", encoding="utf8") as file):
            TRIVIA_CACHE = json.load(file)["Data"]
    return TRIVIA_CACHE


def filler_no_response_tokens_shakespeare(rnd: Random, num_tokens: int, encoding_name="cl100k_base"):
    filler_messages = []
    current_tokens = 0
    encoding = tiktoken.get_encoding(encoding_name)

    with open(DATA_DIR.joinpath("shakespeare/shakespeare.txt"), "rb") as f:
        f.seek(0, 2)
        b = f.tell()
        max_tokens_for_message = num_tokens // 3
        pos = int(b * rnd.random()) - max_tokens_for_message * 4
        f.seek(pos)
        while current_tokens < num_tokens:
            # Approx. 4 chars per token
            to_read = min((num_tokens - current_tokens) * 4, max_tokens_for_message * 4)
            message = f.read(to_read).decode(errors="replace")
            filler_messages.append(message)
            current_tokens += len(encoding.encode(message))

    return filler_messages


def _generate_trivia_content(rnd: Random, num_tokens: int, token_len_function: Callable[[str], int]):
    if num_tokens <= 0:
        return [], []
    data = get_trivia()
    messages = list()
    answers = list()
    token_count = 0
    while True:
        new_messages, new_answers = messages[:], answers[:]
        rnd_state = rnd.getstate()
        for _ in range(max(1, 2 * len(messages))):
            trivia = rnd.choice(data)
            new_messages.append(f"Q: {trivia['Question']}, A: {trivia['AnswerValue']}")
            new_answers.append(trivia["AnswerValue"])
        new_token_count = token_len_function("\n".join(new_messages) + "\n" + json.dumps(new_answers))
        if new_token_count > num_tokens:
            # Restore state to before generating these entries that we didn't use, so that it generates the exact same
            # output as the iterative version.
            rnd.setstate(rnd_state)
            break
        token_count = new_token_count
        messages, answers = new_messages, new_answers
    if len(messages) <= 1:
        return messages, answers
    new_messages, new_answers = _generate_trivia_content(rnd, num_tokens - token_count, token_len_function)
    return messages + new_messages, answers + new_answers


def filler_no_response_tokens_trivia(
    rnd: Random, num_tokens: int, max_message_size: int, token_len_function: Callable[[str], int]
) -> tuple[str, str]:
    intro = (
        "Here are some trivia questions and answers for you to process."
        ' Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2", ...]\n'
    )
    tokens_to_return = min(num_tokens, max_message_size)
    total_tokens = token_len_function(intro)
    messages, answers = _generate_trivia_content(rnd, tokens_to_return - total_tokens, token_len_function)
    if len(messages) == 0:
        trivia = rnd.choice(get_trivia())
        messages.append(f"Q: {trivia['Question']}, A: {trivia['AnswerValue']}")
        answers.append(trivia["AnswerValue"])
    return intro + "\n".join(messages), json.dumps(answers)


def filler_task_characters(rnd: Random, agent: ChatSession, num_characters: int):
    # We need a book of some kind - lets do the complete works of shakespeare
    current_characters = 0
    with open(DATA_DIR.joinpath("shakespeare/shakespeare.txt"), "rb") as f:
        f.seek(0, 2)
        b = f.tell()
        pos = int(b * rnd.random()) - num_characters
        f.seek(pos)
        print("Filler: I am going to give you some passages of information now.")
        response = agent.message_to_agent("I am going to give you some passages of information now.")
        print(f"Agent: {response}")
        max_chars_for_message = num_characters // 3
        while current_characters < num_characters:
            to_read = min(num_characters - current_characters, max_chars_for_message)
            message = f.read(to_read).decode(errors="replace")
            print(f"Filler: {message}")
            response = agent.message_to_agent(message)
            print(f"Agent: {response}")
            current_characters += len(message) + len(response)

    print(f"Filler: The information has ended. Please summarise the above passages for me.")
    response = agent.message_to_agent("The information has ended. Please summarise the above passages for me.")
    print(f"Agent: {response}")
