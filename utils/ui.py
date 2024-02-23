from colorama import Fore, Style


def ask_yesno(info: str = "", question: str = "Do you wish to continue?", default_yes=True) -> bool:
    while True:
        action_str = "[y]/n" if default_yes else "y/[n]"
        action = input(f"{info} {question} {action_str}: ").lower().strip()
        if action in ["y", "n", ""]:
            if default_yes:
                return action != "n"
            else:
                return action == "y"


def colour_print(colour: str, *args, end="\n", sep=" ", **kwargs):
    colour = colour.upper()
    if colour.startswith("LIGHT"):
        colour += "_EX"
    print(getattr(Fore, colour) + sep.join(str(a) for a in args), end=end + Style.RESET_ALL, **kwargs)


def multiline_input(message: str) -> str:
    lines = list()
    print(message, end="")
    prefix = ""
    while True:
        in_str = input(prefix)
        if in_str == "":
            if prefix == "":
                print()
            return "\n".join(lines).strip()
        lines.append(in_str)
        prefix = "> "


def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix
