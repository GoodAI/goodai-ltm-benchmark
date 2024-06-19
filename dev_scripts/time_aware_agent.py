import re
from typing import Optional
from datetime import datetime
from utils.llm import set_api_key
from utils.llm import LLMContext, LLMMessage, make_system_message, make_user_message, make_assistant_message
from utils.ui import multiline_input, colour_print
import litellm
from litellm import completion

MODEL = "gpt-4o"
INPUT_PREFIX = "> "
TARGET_LOOP_SECONDS = 15

system_prompt = """
You are an AI assistant.
You don't have to answer right away. Take your time to think if you need it.
All events are timestamped (YYYY-MM-DD hh:mm:ss). Current time is {current_timestamp}.
You are aware of the passing of time and you can think about future instances of yourself and events that take place in the future.
You don't need to timestamp your responses. Your responses will be timestamped automatically.
If you just want to think out loud, start your response with "THINK".
If you don't feel like saying anything at the moment, just write "IDLE".
""".strip()


def str_timestamp(ts: Optional[datetime] = None) -> str:
    if ts is None:
        ts = datetime.now()
    return str(ts)[:-7]


def tstamp(msg: str, ts: Optional[datetime] = None) -> str:
    return f"{str_timestamp(ts)}\n{msg}"


def dynamic_system_prompt() -> LLMMessage:
    return make_system_message(system_prompt.format(current_timestamp=str_timestamp()))


def ask_llm(context: LLMContext) -> str:
    set_api_key()
    context[0] = dynamic_system_prompt()
    response = completion(
        model=MODEL,
        messages=context,
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    context = [dynamic_system_prompt()]
    while True:
        msg = multiline_input(INPUT_PREFIX)
        if msg == "EXIT":
            return
        elif msg != "":
            context.append(make_user_message(tstamp(msg)))
            colour_print("yellow", context[-1]["content"])
        response = ask_llm(context)
        print(response)
        # 2024-06-14 08:16:02
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\n)", response)
        if m is not None:
            response = response.removeprefix(m.group(1))
        context.append(make_assistant_message(tstamp(response)))
        colour_print("blue", context[-1]["content"])


if __name__ == "__main__":
    main()
