import json
import random

import tiktoken

from model_interfaces.gpt_interface import GPTChatSession
from model_interfaces.interface import ChatSession
from utils.constants import DATA_DIR
from utils.text import token_len


TRIVIA_CACHE = None


def get_trivia():
    global TRIVIA_CACHE
    if TRIVIA_CACHE is None:
        with (open(DATA_DIR.joinpath("trivia/trivia.json"), "r", encoding="utf8") as file):
            TRIVIA_CACHE = json.load(file)["Data"]
    return TRIVIA_CACHE


def filler_no_response_tokens_shakespeare(num_tokens: int, encoding_name="cl100k_base"):
    filler_messages = []
    current_tokens = 0
    encoding = tiktoken.get_encoding(encoding_name)

    with open(DATA_DIR.joinpath("shakespeare/shakespeare.txt"), "rb") as f:
        f.seek(0, 2)
        b = f.tell()
        max_tokens_for_message = num_tokens // 3
        pos = int(b * random.random()) - max_tokens_for_message * 4
        f.seek(pos)
        while current_tokens < num_tokens:
            # Approx. 4 chars per token
            to_read = min((num_tokens - current_tokens) * 4, max_tokens_for_message * 4)
            message = f.read(to_read).decode(errors="replace")
            filler_messages.append(message)
            current_tokens += len(encoding.encode(message))

    return filler_messages


def filler_no_response_tokens_trivia(num_tokens: int, max_message_size: int):
    data = get_trivia()
    message = (
        "Here are some trivia questions and answers for you to process."
        ' Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2", ...]\n'
    )
    tokens_to_return = min(num_tokens, max_message_size)
    total_tokens = token_len(message)
    messages = [message]
    at_least_one_trivia = False

    while not at_least_one_trivia or total_tokens < tokens_to_return:
        trivia = random.choice(data)
        trivia_msg = f"Q: {trivia['Question']}, A: {trivia['AnswerValue']}\n"
        total_tokens += token_len(trivia_msg)
        messages.append(trivia_msg)
        at_least_one_trivia = True

    return "".join(messages)


def filler_task_characters(agent: ChatSession, num_characters: int):
    # We need a book of some kind - lets do the complete works of shakespeare
    current_characters = 0
    with open(DATA_DIR.joinpath("shakespeare/shakespeare.txt"), "rb") as f:
        f.seek(0, 2)
        b = f.tell()
        pos = int(b * random.random()) - num_characters
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


if __name__ == "__main__":
    m = GPTChatSession()
    filler = filler_no_response_tokens_trivia(1000)
    for k in filler:
        print(k)
