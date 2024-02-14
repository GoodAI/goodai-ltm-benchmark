import json
import pyperclip
from datasets.chapterbreak import ChapterBreakDataset
from utils.openai import ask_llm, make_system_message, make_user_message

model = "gpt-3.5-turbo-0125"
ds = ChapterBreakDataset()
samples = ds.get_samples(ds.load_data())
s = samples[1]
llm = lambda ctx: ask_llm(ctx, model, temperature=0, context_length=16384)
last_pages = s["ctx"].replace("’", "'")

#----------------------------------#
# Extract differentiating elements #
#----------------------------------#

# extraction_prompt_template = """
# Take a look at this text:
#
# --- Main Text ---
# {pos}
#
# And compare it to these other texts:
#
# {others}
#
# What are the elements that make the main text unique with respect to the other options? Answer with a list of strings that are exact references to the main text. It must be possible to find the strings in the main text using ctrl-F. Respond with a JSON like this:
# [
#   "one string",
#   "another string",
#   ...
# ]
#
# Respond with an empty list if there is no differentiating element.
# """.strip()
#
#
# others = list()
# for i, txt in enumerate(s["negs"]):
#     others.append(f"--- Text {i + 1} ---\n{txt}")
#
# context = [
#     make_system_message("You are an assistant for finding connections between texts."),
#     make_user_message(extraction_prompt_template.format(
#         pos=s["pos"],
#         others="\n\n".join(others),
#     ))
# ]
#
# print("--- Main Text ---")
# print(s["pos"])
#
# response_elements = llm(context)
# print("\n--- Differentiating elements ---")
# print(response_elements)

response_elements = """
[
  "Taehyung wakes up a little more sluggish than usual",
  "cheeks flushed like he’s feverish",
  "thermometer gives them nothing",
  "weather change was really unexpected",
  "susceptible to them Taehyung is",
  "Taehyung rarely ever gets sick",
  "iron immunity",
  "official hospital visiting hours",
  "Taehyung almost passes out while standing"
]
""".strip()

#------------------------------------#
# Find those elements in the context #
#------------------------------------#
find_elements_system_template = """
You are an assistant for finding hints (even if subtle) in a long text that somewhat anticipate things that appear in future parts of the text.

These are the things that might be anticipated in the text:
{elements}

Answer with a list of very short text excerpts pointing to places where things from the list are somewhat anticipated. It must be possible to find the strings in the text using ctrl-F. Respond with a JSON like this:
[
  {{
    "excerpt": "text piece that you consider that anticipates one of the list items",
    "item": "item from the list that you think the excerpt references to",
    "reasoning": "why do you think that the excerpt anticipates the item",
    "is_valid": "yes or no, for whether you conclude that you have made a good call",
  }},
  ...
]

Respond with an empty list if there are no mentions or references to those things.
"""

find_elements_template = """
This is the text where you have to look for possible anticipations:

{text}
""".strip()

elements = json.loads(response_elements)
elements = "\n".join("- " + el for el in elements)
context = [
    make_system_message(find_elements_system_template.format(elements=elements)),
    make_user_message(find_elements_template.format(
        text=s["ctx"],
    ))
]

model = "gpt-4-turbo-preview"
response_references = llm(context)
print("\n--- References ---")
print(response_references)
d = json.loads(response_references)
for ref in d:
    if not ref["is_valid"]:
        continue
    print(f"Ref: {ref['excerpt']}")
    i = last_pages.find(ref['excerpt'])
    print(f"In text: {i >= 0}")
    if i >= 0:
        print(f"Reasoning: {ref['reasoning']}")
        print(f"Pos: {i}")
