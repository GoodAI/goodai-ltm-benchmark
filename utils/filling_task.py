import json
from random import Random
from typing import Callable

import tiktoken
from model_interfaces.interface import ChatSession
from utils.constants import DATA_DIR


TRIVIA_CACHE = None


def get_trivia():
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


def filler_no_response_tokens_trivia(rnd: Random, num_tokens: int, max_message_size: int, token_len_function: Callable[[str], int]):
    data = get_trivia()
    message = (
        "Here are some trivia questions and answers for you to process."
        ' Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2", ...]\n'
    )
    tokens_to_return = min(num_tokens, max_message_size)
    total_tokens = token_len_function(message)
    messages = [message]
    answers = []
    at_least_one_trivia = False
    est_response_tokens = 0

    while not at_least_one_trivia or (total_tokens + est_response_tokens) < tokens_to_return:
        trivia = rnd.choice(data)
        trivia_msg = f"Q: {trivia['Question']}, A: {trivia['AnswerValue']}\n"
        answers.append(trivia['AnswerValue'])
        total_tokens += token_len_function(trivia_msg)
        est_response_tokens = token_len_function(str(answers))
        messages.append(trivia_msg)
        at_least_one_trivia = True

    return "".join(messages), str(answers)


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
