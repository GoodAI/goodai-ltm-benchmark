from colorama import Fore, Style


def render_context(context):
    for ctx in context:
        if ctx["role"] == "user":
            print(Fore.YELLOW + f"USER:\n{ctx['content']}\n")
        if ctx["role"] == "assistant":
            print(Fore.CYAN + f"ASSISTANT:\n{ctx['content']}\n")

    print(Style.RESET_ALL)


def flatten_context(context):
    string = ""
    for ctx in context:
        string += ctx["content"]

    return string


def search_context(context, content=None, timestamp=None):
    for idx, c in enumerate(context):
        if content and c["content"] != content:
            continue

        if timestamp and c["timestamp"].__str__() != timestamp:
            continue

        return idx

    for c in context:
        c["timestamp"] = str(c["timestamp"])
    raise ValueError(
        "Context search failed!\n"
        f"content = {repr(content)}\n"
        f"timestamp = {timestamp}\n"
        f"history:\n{context}"
    )
