import json
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from random import randint
from typing import List, Callable, Tuple, Optional, Any, Iterator

import tiktoken
from goodai.helpers.json_helper import sanitize_and_parse_json

from utils.context import flatten_context, search_context
from utils.openai import ask_llm
from utils.files import make_testdef_path

# _match_system_prompt = """
# You are to judge whether the provided answers are correct, given question(s) and
# expected information.
#
# Take a deep breath and consider these questions:
# - Does the information in the actual answer accurately represent the expected information?
# - Do any stated quantities match?
# - For any extra information in the actual answer: Does it directly contradict a statement in the expected information? If it doesn't, then the extra information should not be penalized.
#
# All other extra information and context in the actual answer is permitted.
# Disregard spelling errors. Respond in JSON with the following format:
# {
#     "reasoning": List[string], // Your careful reasoning after taking a deep breath
#     "correct": List[int] // Is a provided answer correct? 1 for yes, 0 for no.
# }
# """

_match_system_prompt = """
You are to evaluate some provided answers, given question(s) and
expected answers, plus a checklist that you must follow. For each question, you will go through the following checklist and provide a yes or no answer.

1. Does the answer make reference to the expected answer?
2. Is the expected answer a number or a quantity?
3. If the answer is a number or a quantity, do they match?

Respond in JSON with the following format:
[
  {
    "question_nr": 1,
    "checklist": ["yes", "no", ...],
  },
  {
    "question_nr": 2,
    "checklist": ["no", "no", ...],
  },
  ...
]
"""


class TestAction:
    pass


class TestFinishedAction(TestAction):
    pass


@dataclass
class SendMessageAction(TestAction):
    message: str
    reply: Optional[str] = None
    reply_ts: Optional[datetime] = None
    sent_ts: Optional[datetime] = None
    is_question: bool = False
    is_filling: bool = False


@dataclass
class SendAndRegisterAction(SendMessageAction):
    pass


@dataclass
class WaitAction(TestAction):
    tokens: int = 0
    time: timedelta = field(default_factory=timedelta)


@dataclass
class TestExample:
    dataset_name: str = ""
    description: str = ""
    dataset_generator: "DatasetInterface" = None
    script: List[str] = field(default_factory=list)
    expected_responses: Any = None
    can_be_interleaved: bool = True
    uses_callback: bool = False
    evaluation_fn: Callable[[List[str], list[str], List[Any]], tuple[int, int, List[str]]] = None
    time_jumps: list[timedelta | None] = None
    token_spacings: list[int] = None
    is_temporal: bool = False
    example_id: str = ""
    is_question: List[bool] = field(default_factory=list)
    number_of_questions: int = 0
    finished: bool = False
    _iter: Iterator[TestAction] = None
    reset_message: str = ""

    @property
    def unique_id(self):
        return f"{self.dataset_name} - {self.example_id}"

    def get_path(self, run_name: str) -> Path:
        return make_testdef_path(run_name, self.dataset_name, self.example_id)

    def __post_init__(self):
        self._iter = self.action_iter()
        if self.token_spacings is None:
            self.token_spacings = [0] * len(self.script)
        if self.time_jumps is None:
            self.time_jumps = [timedelta() for _ in self.script]

    def action_iter(self) -> Iterator[TestAction]:
        scripts = [self.script, self.token_spacings, self.time_jumps, self.is_question]
        for msg, tokens, t_jump, is_q in zip(*scripts):
            if self.uses_callback and is_q:
                yield SendAndRegisterAction(msg, is_question=is_q)
            else:
                yield SendMessageAction(msg, is_question=is_q)
            if tokens > 0 or t_jump.total_seconds() > 0:
                yield WaitAction(tokens=tokens, time=t_jump)
        if self.is_temporal and len(self.is_question) == len(self.script) + 1:
            yield SendMessageAction("", is_question=self.is_question[-1])
        self.finished = True

    def step(self) -> TestAction | None:
        assert not self.finished
        try:
            return next(self._iter)
        except StopIteration:
            return None

    def to_dict(self) -> dict:
        return dict(
            script=self.script,
            is_question=self.is_question,
            time_jumps=[td.total_seconds() for td in self.time_jumps],
            token_spacings=self.token_spacings,
            expected_responses=self.expected_responses,
            can_be_interleaved=self.can_be_interleaved,
            evaluation_fn=self.evaluation_fn.__name__,
            is_temporal=self.is_temporal,
            uses_callback=self.uses_callback,
            reset_message=self.reset_message,
        )

    def save(self, run_name: str, exist_ok: bool = False):
        file_path = self.get_path(run_name)
        if not exist_ok:
            assert not file_path.exists(), f"Attempt to overwrite test {file_path}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fd:
            json.dump(self.to_dict(), fd)

    @classmethod
    def load(cls, file_path: Path | str) -> "TestExample":
        from dataset_interfaces.factory import DATASETS_BY_NAME

        file_path = Path(file_path)
        assert file_path.exists(), f"Test file doesn't exist: {file_path}"
        assert file_path.name.endswith(".def.json")
        with open(file_path) as fd:
            d = json.load(fd)
        d["example_id"] = file_path.name.removesuffix(".def.json")
        d["dataset_name"] = file_path.parent.name
        d["dataset_generator"] = None
        d["dataset_generator"] = DATASETS_BY_NAME.get(d["dataset_name"], lambda: None)()
        assert d["dataset_generator"] is not None, f"Couldn't find a generator for dataset {d['dataset_name']}."
        d["description"] = d["dataset_generator"].description
        d["evaluation_fn"] = getattr(d["dataset_generator"], d["evaluation_fn"])
        d["time_jumps"] = [timedelta(seconds=s) for s in d["time_jumps"]]
        d["number_of_questions"] = len([q for q in d["is_question"] if q])
        example_class = CallBackTestExample if d.get("uses_callback", False) else TestExample
        return example_class(**d)


@dataclass
class CallBackTestExample(TestExample):
    def step(self) -> TestAction | None:
        assert not self.finished
        try:
            return next(self._iter)
        except StopIteration:
            self.finished = False
            return WaitAction(tokens=1)


@dataclass
class DatasetInterface(ABC):
    name: str
    description: str
    question: str = ""
    filler_tokens_low: int = 0
    filler_tokens_high: int = 0
    pre_question_filler: int = 0
    seed: int = 0
    cost_callback: Callable[[float], None] = None
    uses_callback: bool = False
    reset_message: str = ""

    def count_questions(self, is_question):
        return len([x for x in is_question if x])

    @abstractmethod
    def generate_examples(self, num_examples: int) -> List[TestExample]:
        pass

    @abstractmethod
    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str]]:
        pass

    def evaluate_correct_gpt(
        self,
        questions: List[str],
        provided_answer: List[str],
        expected_answer: Any,
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt_impl(questions, provided_answer, expected_answer, self.cost_callback)

    # @staticmethod
    # def evaluate_correct_gpt_impl(
    #     questions: List[str],
    #     provided_answer: List[str],
    #     expected_answer: Any,
    #     cost_callback: Callable[[float], Any] = None,
    # ) -> Tuple[int, int, List[str]]:
    #     max_score = len(expected_answer)
    #     questions_str = json.dumps(questions)
    #     expected_str = json.dumps(expected_answer)
    #     provided_str = json.dumps(provided_answer)
    #
    #     score = 0
    #     reasoning = []
    #     for q, e, p in zip(questions, expected_answer, provided_answer):
    #
    #         ctx = [
    #             {
    #                 "role": "system",
    #                 "content": _match_system_prompt,
    #             },
    #             {
    #                 "role": "user",
    #                 "content": f"# Questions: {q}\n\n"
    #                 f"# Expected information: {e}\n\n"
    #                 f"# Provided answers: {p}",
    #             },
    #         ]
    #
    #         response = ask_llm(context=ctx, model="gpt-4-1106-preview", temperature=0.01, cost_callback=cost_callback)
    #         try:
    #             parsed = sanitize_and_parse_json(response)
    #             correct_list = parsed["correct"]
    #             if isinstance(correct_list, int):
    #                 score += int(correct_list)
    #             else:
    #                 score += sum(correct_list)
    #             reasoning.append(parsed["reasoning"][0])
    #
    #         except Exception as e:
    #             reasoning.append("JSON parse error")
    #
    #     # return score, max_score, reasoning

    @staticmethod
    def evaluate_correct_gpt_impl(
            questions: List[str],
            provided_answer: List[str],
            expected_answer: Any,
            cost_callback: Callable[[float], Any] = None,
    ) -> Tuple[int, int, List[str]]:
        max_score = len(expected_answer)
        questions_str = json.dumps(questions)
        expected_str = json.dumps(expected_answer)
        provided_str = json.dumps(provided_answer)

        q_list = []
        for idx, (q, e, p) in enumerate(zip(questions, expected_answer, provided_answer)):
            q_list.append({"question_nr": idx, "question": q, "expected_answer": e, "answer_given": p})

        ctx = [
            {
                "role": "system",
                "content": _match_system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(q_list)
            },
        ]

        response = ask_llm(context=ctx, model="gpt-4-1106-preview", temperature=0.01, cost_callback=cost_callback)
        score = 0
        reasoning = []
        try:
            parsed = sanitize_and_parse_json(response)

            for eval in parsed:
                yes_no_list = eval["checklist"]
                correct = yes_no_list[0].lower() == "yes" # reference
                if yes_no_list[1].lower() == "yes" and yes_no_list[2].lower() == "no":
                    correct = False

                score = score + 1 if correct else score
                if correct:
                    reasoning.append("Checklist correct")
                else:
                    reasoning.append("Checklist Incorrect")

        except Exception as e:
            reasoning.append("JSON parse error")

        return score, max_score, reasoning


    def create_question(self, example: TestExample, statement_times, time_now):
        # Generate the question for temporal questions
        raise NotImplementedError("This dataset is not meant to have temporal questions.")

    @abstractmethod
    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        pass

    def create_filler(self, is_question: list[bool]) -> list[int]:
        def _filler_size(is_q: bool, is_p2q: bool):
            return self.pre_question_filler if is_p2q else (randint(self.filler_tokens_low, self.filler_tokens_high))

        assert len(is_question) >= 1
        is_prior_to_question = is_question[1:] + [False]
        filler = [_filler_size(is_q, is_p2q) for is_q, is_p2q in zip(is_question, is_prior_to_question)]
        filler[-1] = 0
        return filler

    def tokens_to_answer(self, context: List, example: TestExample, timestamps: List):
        encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = num_characters = 0
        # Get most relevant line, and any characters after that statement.
        script_answer_index, answer_end_char = self.answer_statement_idx(example)
        target_timestamp = timestamps[script_answer_index].__str__()
        relevant_line = example.script[script_answer_index]

        # Update tokens and character counts from the script
        num_characters += len(relevant_line[answer_end_char:])
        num_tokens += len(encoding.encode(relevant_line[answer_end_char:]))

        # Where in the history was the statement made?
        # Find that statement in the history suing the content and timestamp and count from there.
        history_idx = search_context(context, relevant_line, target_timestamp) + 1
        countable_history_chunk = flatten_context(context[history_idx:])

        # Now count the tokens and characters since there
        num_characters += len(countable_history_chunk)
        num_tokens += len(encoding.encode(countable_history_chunk))

        return num_characters, num_tokens

    def continual_evaluation_callback(
        self, scheduler, example: TestExample, task_log: List[str]
    ) -> Tuple[int, int, List[str], bool]:
        raise NotImplementedError("This dataset does not have a callback implemented. Use evaluate_correct instead.")
