import logging

import click

from runner.run_benchmark import get_chat_session


@click.command('run-interactive')
@click.option('-a', '--agent', required=True, type=str)
@click.option('-p', '--max-prompt-size', required=False, type=int, default=2000)
def main(agent: str, max_prompt_size: int):
    chat_session = get_chat_session(agent, max_prompt_size)
    while True:
        user_question = input("User: ")
        user_question = user_question.strip()
        if user_question == ".quit":
            logging.warning("Exiting session.")
            break
        elif user_question == ".reset":
            logging.warning("Clearing history and memory.")
            chat_session.reset()
            continue
        response, _, _ = chat_session.message_to_agent(user_question)
        print(f"Assistant: {response}")


if __name__ == '__main__':
    main()
