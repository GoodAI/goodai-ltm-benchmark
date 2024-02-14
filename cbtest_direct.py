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
continuation = s["pos"].replace("’", "'")

#-----------------------------------------------------------------------#
# Find elements in the context that might anticipate the following text #
#-----------------------------------------------------------------------#
find_elements_system_template = """
You are an assistant for finding hints or traces (even if subtle) in a main text that somewhat anticipate things that are part of the continuation.

First, read the main text carefully.

--- Begin of the main text ---
{text}
--- End of the main text ---

Answer with a list of very short text excerpts pointing to places where things from the continuation are somewhat anticipated. It must be possible to find the strings in the text using ctrl-F. Respond with a JSON like this:
[
  {{
    "excerpt": "text piece from the main text that you consider anticipates something from the continuation",
    "anticipates": "thing from the continuation that you think the excerpt anticipates",
    "reasoning": "why do you think that the excerpt anticipates that thing from the continuation",
    "is_valid": "yes or no, for whether you conclude that you have made a good call",
  }},
  ...
]

Respond with an empty list if you found no anticipation at all.
""".strip()

find_elements_user_template = """
Now, take a look at how the main text continues and see if you can see anything being anticipated in the main text.

--- Begin of the continuation ---
{continuation}
--- End of the continuation ---
"""

context = [
    make_system_message(find_elements_system_template.format(
        text=last_pages,
    )),
    make_user_message(find_elements_user_template.format(
        continuation=continuation,
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
    i = last_pages.find(ref['excerpt'])
    if i < 0:
        continue
    print(f"Ref: {ref['excerpt']}")
    print(f"Reasoning: {ref['reasoning']}")
    print(f"Pos: {i}")
