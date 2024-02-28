import json
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from random import randint
from typing import List, Callable, Tuple, Optional, Any, Iterator, Dict

import tiktoken
from goodai.helpers.json_helper import sanitize_and_parse_json

from utils.context import flatten_context, search_context
from utils.openai import ask_llm, LLMContext
from utils.files import make_testdef_path

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
    dataset_name: str = None
    description: str = None
    dataset_generator: "DatasetInterface" = None
    script: List[str] = field(default_factory=list)
    expected_responses: Any = None
    can_be_interleaved: bool = True
    uses_callback: bool = False
    evaluation_fn: Callable[[List[str], list[str], List[Any]], tuple[int, int, List[str]]] = None
    time_jumps: list[timedelta] = None
    token_spacings: list[int] = None
    is_temporal: bool = False
    example_id: str = ""
    is_question: List[bool] = field(default_factory=list)
    number_of_questions: int = 0
    finished: bool = False
    _iter: Iterator[TestAction] = None
    reset_message: str = None

    @property
    def unique_id(self):
        return f"{self.dataset_name} - {self.example_id}"

    def get_path(self, run_name: str) -> Path:
        return make_testdef_path(run_name, self.dataset_name, self.example_id)

    def __post_init__(self):
        assert self.dataset_generator is not None
        assert self.dataset_name is None
        assert self.description is None
        assert self.reset_message is None
        self.dataset_name = self.dataset_generator.name
        self.description = self.dataset_generator.description
        self.reset_message = self.dataset_generator.reset_message
        if self.evaluation_fn is None:
            self.evaluation_fn = self.dataset_generator.evaluate_correct
        self.number_of_questions = len([q for q in self.is_question if q])
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

    def step(self) -> TestAction | None:
        assert not self.finished
        try:
            return next(self._iter)
        except StopIteration:
            self.finished = True
            return None

    def to_dict(self) -> dict:
        return dict(
            script=self.script,
            is_question=self.is_question,
            time_jumps=[td.total_seconds() for td in self.time_jumps],
            token_spacings=self.token_spacings,
            expected_responses=self.expected_responses,
            can_be_interleaved=self.can_be_interleaved,
            is_temporal=self.is_temporal,
            uses_callback=self.uses_callback,
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
        d["time_jumps"] = [timedelta(seconds=s) for s in d["time_jumps"]]
        d["example_id"] = file_path.name.removesuffix(".def.json")
        d["dataset_name"] = file_path.parent.name
        assert d["dataset_name"] in DATASETS_BY_NAME, f"Couldn't find a generator for dataset {d['dataset_name']}."
        dataset_generator = DATASETS_BY_NAME[d["dataset_name"]]()
        generator_attrs = {"dataset_name", "description", "reset_message"}
        d = {k: d[k] for k in d.keys() if k not in generator_attrs}
        return dataset_generator.create_example(**d)


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
class DynamicExample(TestExample):
    cost_callback: Callable[[float], None] = None
    score: int = 0
    max_score: int = 0
    expected_responses: list[str] = field(default_factory=list)
    script: List[str] = field(default_factory=lambda: [""])  # Required for compatibility, but no actual script.

    def __post_init__(self):
        assert self.evaluation_fn is None, "Dynamic examples have their own evaluation function."
        self.evaluation_fn = lambda *args: self.evaluate()
        super().__post_init__()
        assert self.cost_callback is not None, "Dynamic examples require a cost callback."
        assert self.max_score > 0

    def evaluate(self) -> tuple[int, int, list[str]]:
        return self.score, self.max_score, []

    def ask_llm(self, context: LLMContext, temperature: float = 0, max_tokens: int = 256) -> str:
        return ask_llm(
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            cost_callback=self.cost_callback,
        )

    @abstractmethod
    def action_iter(self) -> Iterator[TestAction]:
        pass


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
    max_message_size: int = 1024

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

    @staticmethod
    def evaluate_correct_gpt_impl(
            questions: List[str],
            provided_answer: List[str],
            expected_answer: Any,
            cost_callback: Callable[[float], Any] = None,
    ) -> Tuple[int, int, List[str]]:
        max_score = len(expected_answer)

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

        except Exception:
            reasoning.append("JSON parse error")

        return score, max_score, reasoning

    def create_question(self, example: TestExample, statement_times, time_now):
        # Generate the question for temporal questions
        raise NotImplementedError("This dataset is not meant to have temporal questions.")

    def create_example(self, **kwargs) -> TestExample:
        return (CallBackTestExample if kwargs.get("uses_callback", False) else TestExample)(
            dataset_generator=self,
            **kwargs
        )

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

    def tokens_to_answer(self, test_context: List[Dict[str, Any]], full_context: List[Dict[str, str]], example: TestExample):
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = num_characters = 0

        # Get most relevant line, and any characters after that statement.
        script_answer_index, answer_end_char = self.answer_statement_idx(example)
        relevant_line = example.script[script_answer_index]

        timestamp_idx = search_context(test_context, relevant_line)
        target_timestamp = test_context[timestamp_idx]["timestamp"].__str__()

        # Update tokens and character counts from the script
        num_characters += len(relevant_line[answer_end_char:])
        num_tokens += len(encoding.encode(relevant_line[answer_end_char:]))

        # Where in the history was the statement made?
        # Find that statement in the history suing the content and timestamp and count from there.
        history_idx = search_context(full_context, relevant_line, target_timestamp) + 1
        countable_history_chunk = flatten_context(full_context[history_idx:])

        # Now count the tokens and characters since there
        num_characters += len(countable_history_chunk)
        num_tokens += len(encoding.encode(countable_history_chunk))

        return num_characters, num_tokens

    def continual_evaluation_callback(
        self, scheduler, example: TestExample, task_log: List[str]
    ) -> Tuple[int, int, List[str], bool]:
        raise NotImplementedError("This dataset does not have a callback implemented. Use evaluate_correct instead.")


@dataclass
class DynamicDataset(DatasetInterface, ABC):
    example_cls: type[DynamicExample] = None

    def __post_init__(self):
        assert self.example_cls is not None and issubclass(self.example_cls, DynamicExample)

    def _proxy_cost_callback(self, cost_usd: float) -> None:
        # `self.cost_callback` is not added until all examples are created and the run starts.
        # This proxy method ensures future access to the updated callback function.
        return self.cost_callback(cost_usd)

    def evaluate_correct(self, *args):
        raise NotImplementedError("This method should not be called. Each test example has its own evaluation function.")

    @abstractmethod
    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        pass

    def create_example(self, **kwargs) -> DynamicExample:
        return self.example_cls(
            dataset_generator=self,
            cost_callback=self._proxy_cost_callback,
            **kwargs
        )
