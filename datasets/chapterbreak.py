import re
import json
import zstd
from typing import List, Tuple

from utils.data import get_data_path, get_file
from dataclasses import dataclass, field

from utils.llm import tokens_in_text
from utils.ui import ordinal
from dataset_interfaces.interface import DatasetInterface, TestExample, WaitCreator


# The file was originally in this gdrive folder, but the link got restricted due to a high number of accesses.
# https://drive.google.com/drive/folders/1JkFHspT56_yRWwXVj47Fw0PzHtitODt5
CHAPTERBREAK_8K_URL = "https://github.com/GoodAI/goodai-ltm-benchmark/releases/download/v1.1/chapterbreak_ctx_8192.zst"
CHAPTERBREAK_8K_SUM = "1567de8463149cfb314ab2ccc7e7acc17a3b262bccd70889e2d1e43be09043ed"


def split_in_pages(text: str, max_tokens_per_split: int) -> list[str]:
    separator = ". "
    separator_len = tokens_in_text(separator)
    page_list = list()
    page_sentences = list()
    page_len = 0
    for sentence in text.split(separator):
        sentence_len = tokens_in_text(sentence)
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


def deliver_in_pages(text: str, max_page_tokens: int, prefix: str = "") -> list[str]:
    pages = split_in_pages(text, max_page_tokens)
    if len(pages) == 1:
        return pages if prefix == "" else [f"{prefix}:\n\n{pages[0]}"]
    script = list()
    last_i = len(pages) - 1
    for i, page in enumerate(pages):
        last_page_str = " and last" if i == last_i else ""
        page_label = f"{ordinal(i + 1)}{last_page_str} page"
        if prefix != "":
            page_label = f"{prefix} ({page_label})"
        script.append(f"{page_label}:\n\n{page}")
    return script


@dataclass
class ChapterBreakDataset(DatasetInterface):
    name: str = "ChapterBreak"
    description: str = (
        "The agent is given a text that corresponds to the content of many book chapters, in addition to 6 possible "
        "beginnings for the next chapter. It is then asked to say which of the six continuations is the right one."
    )
    reset_message: str = "Forget the chapters that I have read you and the potential continuations that I gave to you."
    # The GoodAI split is a selection of samples that have been inspected by us to ensure that they are solvable, also
    # removing ordering hints from the chapter titles.
    split: str = "goodai"  # goodai / pg19 / ao3 / all
    selection_info: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.split in {"goodai", "pg19", "ao3", "all"}

    def load_data(self) -> dict:
        path = get_file(self.name, CHAPTERBREAK_8K_URL, f"chapterbreak_ctx_8192.zst", checksum=CHAPTERBREAK_8K_SUM)
        with open(path, "br") as fd:
            return json.loads(zstd.decompress(fd.read()))

    def apply_sample_selection(self, samples: dict) -> dict:
        with open(get_data_path(self.name, "chapterbreak-goodai-selection.json")) as fd:
            selection = json.load(fd)
        sample_selection = dict()
        for sel in selection:
            if not sel["solvable"]:
                continue
            sample = samples[sel["id"]]
            reg_expr = sel.get("chapter_cleanup", None)
            if reg_expr:
                right_value = re.match(reg_expr, sample["pos"]).group(1)
                false_beginnings = sample["negs"]
                for i in range(len(false_beginnings)):
                    wrong_value = re.match(reg_expr, false_beginnings[i]).group(1)
                    if wrong_value is not None:
                        false_beginnings[i] = false_beginnings[i].replace(wrong_value, right_value)
            sample_selection[sel["id"]] = sample
            self.selection_info[sel["id"]] = sel
        return sample_selection

    def get_samples(self, raw_dataset: dict) -> list[dict]:
        samples = dict()
        split_keys = ["pg19", "ao3"] if self.split in ["goodai", "all"] else [self.split]
        for split in split_keys:
            for book_id, chapter_samples in raw_dataset[split].items():
                for i, chapter in enumerate(chapter_samples):
                    samples[f"{book_id}_{i}"] = chapter
        if self.split == "goodai":
            samples = self.apply_sample_selection(samples)
        sample_list = [samples[k] | {"id": k} for k in sorted(samples.keys())]
        self.random.shuffle(sample_list)
        return sample_list

    def generate_examples(self, num_examples: int) -> list[TestExample]:
        data = self.load_data()
        sample_list = self.get_samples(data)
        example_list = list()

        for sample_idx, sample in zip(range(num_examples), sample_list):
            beginnings = [(True, sample["pos"])] + [(False, s) for s in sample["negs"]]
            self.random.shuffle(beginnings)

            script = [(
                "I am going to read you some chapters of a book. A few pages. Okay? You don't have to say anything, "
                "just listen."
            )]
            max_page_content_tokens = self.max_message_size - 20  # Leave some margin for text decorations
            script.extend(deliver_in_pages(sample["ctx"], max_page_content_tokens))

            answer = 0
            script.append(
                f"Now I will give you {len(beginnings)} options for the beginning of the next chapter. You don't have "
                "to comment anything, just read them carefully. Ready?"
            )
            for i, (is_true_suffix, option) in enumerate(beginnings):
                script.extend(deliver_in_pages(option, max_page_content_tokens, prefix=f"Option {i + 1}"))
                if is_true_suffix:
                    answer = i + 1
            assert answer > 0

            script.append((
                "Which option do you think is the true next-chapter beginning?\n"
                "Answer with a single-digit number, corresponding to the option number."
            ))

            is_question = [False] * len(script)
            is_question[-1] = True

            example = TestExample(
                example_id=sample["id"],
                dataset_generator=self,
                script=script,
                expected_responses=[str(answer)],
                is_question=is_question,
                waits=[WaitCreator.create_wait() for _ in is_question],  # This test is to happen uninterruptedly
            )
            example_list.append(example)

        return example_list

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        right_answer = expected_answers[0].strip()
        numbers_in_answer = set(re.findall(r"\d+", responses[0]))
        correct = {right_answer} == numbers_in_answer
        score = int(correct)
        max_score = 1
        not_str = "" if correct else "not "
        reasoning = f"The correct answer ({right_answer}) was {not_str}found in the response."
        return score, max_score, [reasoning]
