import json
from datasets.chapterbreak import ChapterBreakDataset
from utils.llm import make_system_message, make_user_message, make_assistant_message, LLMContext, ask_llm
from utils.ui import colour_print
from random import Random
from goodai.helpers.json_helper import sanitize_and_parse_json


system_prompt_template = """
You are an AI agent.

TASK DESCRIPTION:
You are reading a document containing chapters of a fan-fiction book, plus six options for the beginning of the following chapter.
Your goal is to decide which option is the real continuation.
Provide your answer as a JSON list with a single integer: [<option_nr>]
""".strip()

full_text_template = """
--- BOOK CHAPTERS ---
{chapters}

--- OPTIONS ---
{options}

--- END ---
""".strip()


def main():

    # Load and prepare ChapterBreak sample
    cbd = ChapterBreakDataset()
    cbd.random = Random(0)
    sample = cbd.get_samples(cbd.load_data())[2]
    options = sample["negs"]
    true_idx = 5
    options.insert(true_idx, sample["pos"])
    full_text = full_text_template.format(
        chapters=sample["ctx"],
        options="\n\n".join([f"Option {i + 1}:\n{opt}" for i, opt in enumerate(options)]),
    )

    model = "gpt-4o"
    context = [
        make_system_message(system_prompt_template),
        make_user_message(full_text),
    ]
    response = ask_llm(context, model, temperature=0.1, max_response_tokens=1024)
    ans = sanitize_and_parse_json(response)[0] - 1
    colour, word = ("green", "RIGHT") if ans == true_idx else ("red", "WRONG")
    colour_print(colour, f"Answer {ans + 1} is {word}")


if __name__ == "__main__":
    main()
