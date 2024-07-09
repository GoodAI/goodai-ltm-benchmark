import json
import pyperclip
from datasets.chapterbreak import ChapterBreakDataset, split_in_pages
from utils.llm import ask_llm, make_system_message, make_user_message


total_cost = 0
# model = "gpt-3.5-turbo-0125"
model = "gpt-4-turbo-preview"
ds = ChapterBreakDataset(split="ao3")
samples = ds.get_samples(ds.load_data())
s = samples[1]
llm = lambda ctx: ask_llm(ctx, model, temperature=0, max_overall_tokens=16384, cost_callback=cost_callback, max_response_tokens=1024)
fix_commas = lambda txt: txt.replace("’", "'")
last_pages = fix_commas(s["ctx"])
continuation = fix_commas(s["pos"])
negs = [fix_commas(n) for n in s["negs"]]
pyperclip.copy("\n-------------\n".join([last_pages, continuation] + negs))
# exit(0)  # Uncomment to just copy the sample to the clipboard


def cost_callback(llm_cost: float):
    global total_cost
    total_cost += llm_cost


def parse(s: str, sep: str = "[]") -> list | dict:
    try:
        i = s.find(sep[0])
        j = s.rfind(sep[1])
        return json.loads(s[i:j + 1])
    except json.decoder.JSONDecodeError:
        return eval(sep)


def parse_list(s: str) -> list:
    return parse(s, "[]")


def parse_dict(s: str) -> dict:
    return parse(s, "{}")


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

Respond with an empty list if there are no differentiating elements.
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
all_negs = "\n".join(negs)
response_elements = llm(context)
elements = parse_list(response_elements)
elements = [e for e in set(elements) if e not in all_negs]
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

And you have identified the following elements as unique and differentiating elements of the text:

{elements}

You will be given a text extracted from previous chapters. Your job is to find sentences in the text provided that might refer to any of these differentiating elements.
Provide (rather longer than shorter) verbatim sentences that can be found via ctrl-F in the provided text. Answer in JSON form like this:
[
  "first verbatim reference",
  "second verbatim reference",
  ...
]
""".strip()

find_elements_user_template = """
Find potential references in this text:

{text}
""".strip()

ctx_pages = split_in_pages(last_pages, 4096)

references = list()
for page in ctx_pages:
    context = [
        make_system_message(find_elements_system_template.format(
            continuation=continuation,
            elements=elements,
        )),
        make_user_message(find_elements_user_template.format(text=page)),
    ]
    response_references = llm(context)
    references.extend(parse_list(response_references))

references = [r for r in set(references) if r in last_pages]

print("\n--- References ---")
print("\n".join("- " + r for r in references))

filter_references_system_template = """
You are an assistant for advanced text analysis.
You are evaluating potential references to this text:

{continuation}

Concretely, references that might connect to the following distinctive elements in the text:

{elements}

You will be given a contextualized sentence to evaluate. You will provide a detailed analysis in JSON form, like this:
{{
  "does the sentence make reference to any of the distinctive elements?": true or false,
  "if true, what element?": "the corresponding distinctive element",
  "if true, is the reference ambiguous?": true or false,
}}
""".strip()

filter_references_user_template = """
Sentence:
{sentence}

Context:
{context}
""".strip()

print("\n--- Selected ---")

context_span = 1000
for r in references:
    i = last_pages.find(r)
    j = i + len(r) + context_span
    i = max(i - context_span, 0)
    context = [
        make_system_message(filter_references_system_template.format(
            continuation=continuation,
            elements=elements,
        )),
        make_user_message(filter_references_user_template.format(
            sentence=r,
            context=last_pages[i:j],
        )),
    ]
    response_filter = llm(context)
    d = parse_dict(response_filter)
    if d["does the sentence make reference to any of the distinctive elements?"]:
        elem = d["if true, what element?"]
        if elem in elements and not d["if true, is the reference ambiguous?"]:
            print(f"- {r}")
            print(f"  ({elem})")

print("Final cost: ", total_cost)
