import json
import random
from inspect import getsource
from typing import List, Tuple

from utils.data import get_gdrive_file
from dataclasses import dataclass
from utils.ui import ordinal
from utils.tokens import token_len
from dataset_interfaces.interface import DatasetInterface, TestExample


# Extracted from gdrive folder
# https://drive.google.com/drive/folders/1JkFHspT56_yRWwXVj47Fw0PzHtitODt5
GDRIVE_8K_ID = "15AcGiC4wIglru2gK2MHSX5Fie7gYxTTS"

extraction_prompt_template = """
You are an expert language analyst who helps in finding connections between different pieces of text within the same document. As input, you will receive some text, which corresponds to the final pages of a chapter, and you will analyze its content in order to find elements that point to the true next-chapter begin as the only option.

Respond with a JSON listing the elements like this:
[
  {
    "reason": "why the element links to the true next-chapter beginning and to no other chapter-beginning",
    "element": "exact string that can be found in the text",
  },
  ...
]

Respond with an empty list if you find no unique connection.

You know that the true next-chapter beginning is this:
{ctx}

And you also know that there exist these other chapter beginnings, which are not from the next chapter, despite being disguised as such:
""".strip()


def split_in_pages(text: str, max_tokens_per_split: int) -> list[str]:
    separator = ". "
    separator_len = token_len(separator)
    page_list = list()
    page_sentences = list()
    page_len = 0
    for sentence in text.split(separator):
        sentence_len = token_len(sentence)
        if page_len == 0:
            page_sentences.append(sentence)
            page_len += sentence_len
            continue
        if page_len + separator_len + sentence_len > max_tokens_per_split:
            page_list.append(separator.join(page_sentences))
            page_sentences.clear()
            page_len = 0
            continue
        page_sentences.append(sentence)
        page_len += separator_len + sentence_len
    if page_len > 0:
        page_list.append(separator.join(page_sentences))
    return page_list


def deliver_in_pages(text: str, max_page_tokens: int, first_page_prefix: str = "") -> list[str]:
    script = list()
    pages = split_in_pages(text, max_page_tokens)
    last_i = len(pages) - 1
    for i, page in enumerate(pages):
        last_page_str = " and last" if i == last_i else ""
        script.append(f"{first_page_prefix}{ordinal(i + 1)}{last_page_str} page:\n\n{page}")
    return script


@dataclass
class ChapterBreakDataset(DatasetInterface):
    name: str = "ChapterBreak"
    description: str = (
        "The agent is given a text that corresponds to the final part of a book "
        "chapter, plus 6 possible continuations. It is then asked to say which of the "
        "six continuations is the right one."
    )
    split: str = "ao3"  # pg19 / ao3 / all
    page_tokens: int = 1024 - 20  # Leave some margin for text decorations
    reset_message: str = "Forget the current chapter and its potential continuations."

    def __post_init__(self):
        assert self.split in {"pg19", "ao3", "all"}

    def load_data(self) -> dict:
        filename = f"chapterbreak_ctx_8192.json"
        path = get_gdrive_file(self.name, GDRIVE_8K_ID, filename)
        with open(path) as fd:
            return json.load(fd)

    def get_samples(self, raw_dataset: dict) -> list[dict]:
        samples = dict()
        split_keys = ["pg19", "ao3"] if self.split == "all" else [self.split]
        for split in split_keys:
            for book_id, chapter_samples in raw_dataset[split].items():
                for i, chapter in enumerate(chapter_samples):
                    samples[f"{book_id}_{i}"] = chapter
        sample_list = [samples[k] | {"id": k} for k in sorted(samples.keys())]
        random.Random(self.seed).shuffle(sample_list)
        return sample_list

    def generate_examples(self, num_examples: int) -> list[TestExample]:
        data = self.load_data()
        sample_list = self.get_samples(data)
        example_list = list()

        for sample_idx, sample in zip(range(num_examples), sample_list):
            beginnings = [(True, sample["pos"])] + [(False, s) for s in sample["negs"]]
            random.Random(self.seed + sample_idx).shuffle(beginnings)

            script = ["I am going to read you the final pages of a book chapter. Okay?"]
            script.extend(deliver_in_pages(sample["ctx"], self.page_tokens))

            answer = 0
            script.append(f"Now I will give you {len(beginnings)} options for the beginning of the next chapter. Ready?")
            for i, (is_true_suffix, option) in enumerate(beginnings):
                script.extend(deliver_in_pages(option, self.page_tokens, first_page_prefix=f"Option {i + 1}. "))
                if is_true_suffix:
                    answer = i + 1
            assert answer > 0

            script.append((
                "Which option is the true next-chapter beginning?\n"
                "Answer with a single-digit number, corresponding to the option number."
            ))

            is_question = [False] * len(script)
            is_question[-1] = True

            example = TestExample(
                dataset_name=self.name,
                description=self.description,
                example_id=sample["id"],
                dataset_generator=self,
                script=script,
                expected_responses=[str(answer)],
                evaluation_fn=self.evaluate_correct,
                is_question=is_question,
                number_of_questions=1,
                reset_message=self.reset_message,
            )
            example_list.append(example)

        return example_list

    def answer_statement_idx(self, example: TestExample) -> tuple[int, int]:
        # TODO: try to figure out where the relevant information actually is
        # TODO: load a sample from the dataset in the Python terminal, and see if I can have GPT help me extract key info
        # TODO: from the gold answer and then find it in the last pages of the chapter.
        # For now, we'll just assume that the whole chapter ending is relevant.
        last_page_idx = None
        for i, script_line in enumerate(example.script):
            if script_line.endswith("options for the beginning of the next chapter. Ready?"):
                last_page_idx = i - 1
                break
        assert last_page_idx is not None
        script_answer_index = last_page_idx
        answer_end_char = len(example.script[last_page_idx])
        return script_answer_index, answer_end_char

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        reasoning = getsource(ChapterBreakDataset.evaluate_correct)
        right_answer = expected_answers[0].strip()
        wrong_answers = [str(i + 1) for i in range(6) if str(i + 1) != right_answer]
        correct = right_answer in responses[0]
        for wrong_ans in wrong_answers:
            if wrong_ans in responses[0]:
                correct = False
                break
        score = int(correct)
        max_score = 1
        return score, max_score, [reasoning]
