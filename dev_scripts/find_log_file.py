from pathlib import Path
from utils.llm import _llm_debug_dir
from utils.text import rouge_l

# Prefix search is usually easier, but sometimes there are hidden characters that difficult the process.
# If that's the case, copy the entire response in `search` and select "rouge".
search_in = ["user", "response"][1]
search_type = ["prefix", "rouge"][0]
search = """
Naomh, would you like to try one of the appetizers like the Classic Caesar Salad or the Spinach and Artichoke Dip?
""".strip()

agent_name = "2024-08-05 11:20:50 - Dev Benchmark 3 32k - 1 Example/gpt-4o-mini-8192-1024"


def get_user_reply_content(p: Path) -> tuple[str, str]:
    user_lines = list()
    response_lines = list()
    in_user = in_response = False
    with open(p) as fd:
        for line in fd:
            line = line.removesuffix("\n")
            if line == "--- Response:":
                in_user = False
                in_response = True
                response_lines.clear()
            elif line == "--- USER":
                in_user = True
                in_response = False
                user_lines.clear()
            elif line == "--- ASSISTANT":
                in_user = in_response = False
            elif in_user:
                user_lines.append(line)
            elif in_response:
                response_lines.append(line)
    return "\n".join(user_lines), "\n".join(response_lines)


def main():
    for p in sorted(_llm_debug_dir.joinpath(agent_name).glob("*.txt")):
        query, response = get_user_reply_content(p)
        text = query if search_in == "user" else response
        if search_type == "prefix":
            if text.startswith(search):
                print(p.name)
                return
        else:
            if rouge_l(search, text) > 0.9:
                print(p.name)
                return
    print("Couldn't find the log file.")


if __name__ == "__main__":
    main()
