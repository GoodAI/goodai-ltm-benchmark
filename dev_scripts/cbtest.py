import json
import pyperclip
from datasets.chapterbreak import ChapterBreakDataset, split_in_pages
from utils.llm import ask_llm, make_system_message, make_user_message
from utils.text import token_len


total_cost = 0
model = "gpt-3.5-turbo-0125"
ds = ChapterBreakDataset(split="ao3")
samples = ds.get_samples(ds.load_data())
s = samples[1]
llm = lambda ctx: ask_llm(ctx, model, temperature=0, max_overall_tokens=16384, cost_callback=cost_callback)
fix_commas = lambda txt: txt.replace("’", "'")
last_pages = fix_commas(s["ctx"])
continuation = fix_commas(s["pos"])
negs = [fix_commas(n) for n in s["negs"]]
pyperclip.copy("\n-------------\n".join([last_pages, continuation] + negs))
# exit(0)  # Uncomment to just copy the sample to the clipboard


def cost_callback(llm_cost: float):
    global total_cost
    total_cost += llm_cost


def parse_list(s: str) -> list:
    i = s.find("[")
    j = s.rfind("]")
    return json.loads(s[i:j + 1])


#----------------------------------#
# Extract differentiating elements #
#----------------------------------#
extraction_prompt_template = """
Take a look at this text:

--- Main Text ---
{pos}

And compare it to these other texts:

{others}

What are the elements that make the main text unique with respect to the other options? Answer with a list of strings that are exact references to the main text. It must be possible to find the strings in the main text using ctrl-F. Respond with a JSON like this:
[
  "one string",
  "another string",
  ...
]

Respond with an empty list if there is no differentiating element.
""".strip()


others = list()
for i, txt in enumerate(negs):
    others.append(f"--- Text {i + 1} ---\n{txt}")

context = [
    make_system_message("You are an assistant for finding connections between texts."),
    make_user_message(extraction_prompt_template.format(
        pos=continuation,
        others="\n\n".join(others),
    ))
]

print("--- True Continuation ---")
print(continuation)

response_elements = llm(context)
elements = parse_list(response_elements)
elements = "\n".join("- " + el for el in elements)
print("\n--- Differentiating elements ---")
print(elements)

#------------------------------------#
# Find those elements in the context #
#------------------------------------#
find_elements_system_template = """
You are an expert in literature.

You have read the following text:

{continuation}

And you have identified the following elements as unique elements of the text, which uniquely identify the text that you have read:

{elements}

You will be given a text extracted from previous chapters. Your job is to find clear and unambiguous references to these unique elements. A reference is clear and unambiguous when it is obvious that it refers to one of the unique elements and it could not refer to anything else.

Provide the references in JSON form, like this:
[
  {{
    "reference": "verbatim text that is selected for making a clear and unambiguous reference to one or more unique elements",
    "context": "larger context in which the reference occurs",
    "connection": "what unique element does this reference point to? In what sense?",
    "ambiguity": "Could the selected text make reference to something else?",
    "reasoning": "Is this an unambiguous reference? Is it clear that it points to one or more unique elements?",
    "select": true or false,  # In retrospective, is this a clear and unambiguous reference?
  }},
  ...
]
""".strip()

# model = "gpt-4-turbo-preview"
ctx_pages = split_in_pages(last_pages, 1024)
references = list()

for page in ctx_pages:
    context = [
        make_system_message(find_elements_system_template.format(
            continuation=continuation,
            elements=elements,
        )),
        make_user_message(page)
    ]
    response_references = llm(context)
    references.extend(parse_list(response_references))

references = [r for r in references if r["select"] and r["reference"] in last_pages]

print("\n--- References ---")
for i, r in enumerate(references):
    print(f"Ref. {i + 1}")
    for k in ["reference", "context", "reasoning", "connection"]:
        print(f"{k}: {r[k]}")

print("Final cost: ", total_cost)
