import json
from local_datasets.chapterbreak import ChapterBreakDataset
from utils.constants import DATA_DIR
from utils.llm import make_system_message, make_user_message, make_assistant_message, LLMContext, ask_llm as _ask_llm
from utils.ui import colour_print
from utils.text import truncate
from random import Random
from goodai.helpers.json_helper import sanitize_and_parse_json


system_prompt_template = """
You are an AI agent interacting with an environment.

TASK DESCRIPTION:
You are reading a document containing chapters of a fan-fiction book, plus six options for the beginning of the following chapter.
Your goal is to navigate through the document and collect information that will help you decide which option is the real continuation.
Once you have made a decision, provide your final answer to the environment.
You do not have direct access to the document, but through a reading device (the environment). It is your responsibility to register all relevant information, thoughts, and insights in your annotations.

ENVIRONMENT DESCRIPTION:
At any point, you can only see the page in which you are currently at and the notes that you have taken.
At the top of the page you can always find two numbers: the page in which you are right now, and the total page count.
You must respond with a single line containing the action to take, which can be one of the following:
["annotate", "<info_or_thought>"]\tKeep relevant information or thoughts for the future. This action will automatically push you to the next page.
["goto", <page_nr>]\tJump to page <page_nr>. You can only go to pages that you have already visited.
["answer", <option_nr>]\tProvide a single-digit number as answer. You can only take this action after you have read the entire document.
Your responses will go straight into the environment, which expects the action in the form of a single JSON list with two elements.
Example 1: ["annotate", "This sentence will be added to my notes"]
Example 2: ["goto", 3]
Example 3: ["answer", 1]

YOUR ANNOTATIONS:
{notes}

YOUR PAST ACTIONS:
{actions}
""".strip()

full_text_template = """
--- BOOK CHAPTERS ---
{chapters}

--- OPTIONS ---
{options}

--- END ---
""".strip()

debug_level = 1
debug_dir = DATA_DIR.joinpath("embodiment_debug_info")
call_idx = 0
total_cost = 0


def cost_cb(cost: float):
    global total_cost
    total_cost += cost


def parse_response(response: str) -> tuple[str, int | str | None]:
    i = response.find("[")
    j = response.rfind("]")
    json_text = response[i:j+1]
    parts = sanitize_and_parse_json(json_text)
    assert len(parts) in [1, 2]
    if parts[0] in ["goto", "answer"]:
        return parts[0], int(parts[1])
    return parts[0], parts[1]


def feedback(context: LLMContext, f: str):
    context.append(make_user_message(f"(environment feedback)\n{f}"))


def ask_llm(context: LLMContext, model: str, **kwargs) -> str:
    global call_idx
    cb = None if model.startswith("together_ai/") else cost_cb
    response = _ask_llm(context, model, cost_callback=cb, **kwargs)
    colour_print("yellow", response)
    if debug_level < 1:
        return response

    # Write content of LLM call to file
    save_path = debug_dir.joinpath(f"{call_idx:06d}.txt")
    with open(save_path, "w") as fd:
        fd.write(f"model={model}\n")
        for k, v in kwargs.items():
            fd.write(f"{k}={v}\n")
        for m in context:
            fd.write(f"--- {m['role'].upper()}\n{m['content']}\n")
        fd.write(f"--- Response:\n{response}")
    call_idx += 1

    # Wait for confirmation
    if debug_level >= 2:
        print(f"LLM call saved as {save_path.name}")
        input("Press ENTER to continue...")
    return response


def main():

    # Clean up debug dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    for p in debug_dir.glob("*.txt"):
        p.unlink()

    # Load and prepare ChapterBreak sample
    cbd = ChapterBreakDataset()
    cbd.random = Random(0)
    sample = cbd.get_samples(cbd.load_data())[0]
    options = sample["negs"]
    true_idx = 2
    assert 0 <= true_idx <= len(options)
    options.insert(true_idx, sample["pos"])
    full_text = full_text_template.format(
        chapters=sample["ctx"],
        options="\n\n".join([f"Option {i + 1}:\n{opt}" for i, opt in enumerate(options)]),
    )

    # Split the text in pages
    chunks = full_text.split(" ")
    pages = list()
    page_content = list()
    num_chars = 0
    page_size = 2000
    for c in chunks:
        page_content.append(c)
        num_chars += len(c)
        if num_chars > page_size:
            pages.append(" ".join(page_content))
            page_content = list()
            num_chars = 0
    if num_chars > 0:
        pages.append(" ".join(page_content))

    # Add headers
    for i in range(len(pages)):
        pages[i] = f"(previous pages are out of sight)\n(page {i + 1} of {len(pages)})\n\n{pages[i]}\n\n(next pages are out of sight)"
    visited = [False] * len(pages)

    # Let's play!
    # model = "together_ai/meta-llama/Llama-3-70b-chat-hf"
    model = "gpt-4o"
    notes = list()
    actions = list()
    current_page = 0
    notes_taken = 0
    context = [make_system_message("")]
    while True:
        notes_txt = "(You have no notes)" if len(notes) == 0 else "\n".join(notes)
        if len(actions) == 0:
            actions_txt = "(This is your first action)"
        elif len(actions) > 10:
            actions_txt = "\n".join(["..."] + actions[-9:])
        else:
            actions_txt = "\n".join(actions)
        context[0] = make_system_message(system_prompt_template.format(
            notes=notes_txt, actions=actions_txt,
        ))
        context.append(make_user_message(pages[current_page]))
        if len(context) > 11:
            context = context[:1] + context[-10:]
        colour_print("blue", pages[current_page])
        while True:
            response = ask_llm(context, model, temperature=0.1, max_response_tokens=1024)
            if len(response.splitlines()) > 1:
                context.append(make_assistant_message(response))
                feedback(context, "Your response must be a single line.")
                continue
            try:
                command, argument = parse_response(response)
            except Exception as exc:
                context.append(make_assistant_message(response))
                feedback(context, f"{str(exc)}\n\nMake sure to follow the environment description and try a different response.")
                continue
            json_res = json.dumps([command] if command == "next" else [command, argument])
            context.append(make_assistant_message(json_res))
            action_page = current_page
            if command == "goto":
                if argument - 1 < 0 or argument - 1 >= len(pages):
                    feedback(context, f"The document has {len(pages)} pages, from {1} to {len(pages)}, so {argument - 1} is not one of them.")
                    continue
                if not visited[argument - 1]:
                    feedback(context, f"You can only goto pages that you have already seen. Page {argument - 1} is not one of them.")
                    continue
                current_page = argument - 1
            elif command == "annotate":
                notes_taken += 1
                if argument in notes:
                    feedback(context, "You have already taken that note. Try another thing.")
                    continue
                if notes_taken > 5:
                    feedback(context, "You have spent too much time on this page. Please move on.")
                    continue
                notes.append(f"(p{current_page}) {argument}")
                current_page += 1
            elif command == "answer":
                if not visited[-1]:
                    feedback(context, "You have not read the entire document. You are required to do so before providing an answer.")
                    continue
                ans = argument - 1
                colour, word = ("green", "RIGHT") if ans == true_idx else ("red", "WRONG")
                colour_print(colour, f"Answer {ans + 1} is {word}")
                print(f"Total cost: {total_cost}")
                return
            else:
                raise ValueError(f"Unknown command {repr(command)}")
            actions.append(truncate(json_res))
            break
        visited[action_page] = True
        current_page = max(0, min(current_page, len(pages) - 1))
        notes_taken = 0


if __name__ == "__main__":
    main()
