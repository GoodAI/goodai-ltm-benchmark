from typing import Optional, Callable
from utils.openai import (
    LLMContext,
    ask_llm,
    make_message,
    system_message,
    make_user_message,
    make_assistant_message,
    context_token_len,
)


def setup_context(context: LLMContext, topic: str) -> LLMContext:
    context = [
        make_message(m["role"], m["content"])
        for m in context
        if m["role"] in {"user", "assistant"}
    ]
    system_prompt = (
        # "You are a conversational agent.\n"
        "You pretend to be a human holding a conversation with another human.\n"
        "Your main goal is to keep the conversation going.\n"
        f"Your secondary goal is described as follows: {topic}"
        if topic != ""
        else ""
    )
    return [system_message(system_prompt)] + context


def switch_roles(context: LLMContext) -> LLMContext:
    switch = dict(system="system", user="assistant", assistant="user")
    return [make_message(switch[m["role"]], m["content"]) for m in context]


def phil(
    context: LLMContext,
    topic: str,
    agent_wrapper: Callable[[LLMContext], str],
    model: Optional[str] = None,
    until_tokens: Optional[int] = None,
    until_messages: Optional[int] = None,
    debug: Optional[bool] = False,
) -> LLMContext:
    assert until_tokens or until_messages
    context = setup_context(context, topic)
    init_context_len = len(context)
    if debug and until_messages:
        print(f"Running phil for {until_messages} messages.")
    if debug and until_tokens:
        print(f"Running phil for {until_tokens} tokens.")
    while True:
        previous_role = context[-1]["role"]
        if previous_role == "user":
            agent_response = agent_wrapper(context)
            if debug:
                print(f"Assistant response: {agent_response}")
            context.append(make_assistant_message(agent_response))
        else:
            phil_response = ask_llm(switch_roles(context), model=model)
            if debug:
                print(f"User response: {phil_response}")
            context.append(make_user_message(phil_response))
        added_context = context[init_context_len:]
        if (
            until_tokens
            and context_token_len(added_context, model=model) >= until_tokens
        ):
            break
        if until_messages and len(added_context) >= until_messages:
            break
    return added_context


if __name__ == "__main__":

    def main():
        context = [
            {"role": "user", "content": "Hi there."},
            {
                "role": "assistant",
                "content": "Hello! It's great to see you. How can I assist you today?",
            },
            {
                "role": "user",
                "content": 'I would like to talk about books, if you agree. And once we finish talking about books, I would like you to recall this quote and mention it to me: "In the garden of life, every flower has its season to bloom." - Elara Moonfield\nShall we start?',
            },
        ]

        def agent_wrapper(context):
            msg = context[-1]
            print(msg["role"].upper())
            for line in msg["content"].splitlines():
                print("  " + line)
            print("\n")
            print("ASSISTANT")
            agent_lines = list()
            while True:
                line = input(". ").strip()
                if line == "":
                    break
                agent_lines.append(line)
            return "\n".join(agent_lines)

        topic = "Talk about books and only about things related to books"
        phil(context, topic, agent_wrapper, until_messages=3)
        print("--- Phil has finished filling ---")

    main()
