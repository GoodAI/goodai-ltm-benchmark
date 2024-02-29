import json
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from random import randint
from typing import List, Callable, Tuple, Optional, Any, Iterator, Dict

import tiktoken
from goodai.helpers.json_helper import sanitize_and_parse_json

from utils.constants import DATA_DIR
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
    percentage_finished: float = 0.0


class WaitCreator:
    @classmethod
    def create_wait(cls, tokens: int = 0, time: timedelta = timedelta(seconds=0), percentage_finished: float = 0.0):
        return {"tokens": tokens, "time": time, "percentage_finished": percentage_finished}

    @classmethod
    def unserialise(cls, w_dict):
        return {"tokens": w_dict['tokens'], "time": timedelta(w_dict['time']), "percentage_finished": w_dict['percentage_finished']}

    @classmethod
    def serialise(cls, w_dict):
        return {"tokens": w_dict['tokens'], "time": w_dict['time'].seconds, "percentage_finished": w_dict['percentage_finished']}


@dataclass
class TestExample:
    dataset_generator: "DatasetInterface" = None
    script: List[str] = field(default_factory=list)
    expected_responses: Any = None
    can_be_interleaved: bool = True
    uses_callback: bool = False
    is_temporal: bool = False
    example_id: str = ""
    is_question: List[bool] = field(default_factory=list)
    number_of_questions: int = 0
    finished: bool = False
    _iter: Iterator[TestAction] = None
    waits: List[dict] = field(default_factory=list)

    @property
    def dataset_name(self) -> str:
        return self.dataset_generator.name

    @property
    def description(self) -> str:
        return self.dataset_generator.description

    @property
    def reset_message(self) -> str:
        return self.dataset_generator.reset_message

    @property
    def evaluation_fn(self) -> Callable[[List[str], list[str], List[Any]], tuple[int, int, List[str]]]:
        return self.dataset_generator.evaluate_correct

    @property
    def unique_id(self):
        return f"{self.dataset_name} - {self.example_id}"

    def get_path(self, run_name: str) -> Path:
        return make_testdef_path(run_name, self.dataset_name, self.example_id)

    def __post_init__(self):
        assert self.dataset_generator is not None
        self.number_of_questions = len([q for q in self.is_question if q])
        self._iter = self.action_iter()
        self.waits = self.dataset_generator.default_waits(self.is_question, self.waits)

    def action_iter(self) -> Iterator[TestAction]:
        scripts = [self.script, self.waits, self.is_question]
        for msg, wait, is_q in zip(*scripts):
            if self.uses_callback and is_q:
                yield SendAndRegisterAction(msg, is_question=is_q)
            else:
                yield SendMessageAction(msg, is_question=is_q)
            if wait['tokens'] > 0 or wait['time'].seconds > 0 or wait['percentage_finished'] > 0:
                yield WaitAction(tokens=wait['tokens'], time=wait['time'], percentage_finished=wait['percentage_finished'])
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
            waits=[WaitCreator.serialise(x) for x in self.waits],
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
        d["waits"] = [WaitCreator.unserialise(x) for x in d["waits"]]
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
    reasoning: list[str] = field(default_factory=list)
    script: List[str] = field(default_factory=lambda: [])  # Updated dynamically by `say`
    filler_tokens_low: int = 0
    filler_tokens_high: int = 0
    action: SendMessageAction = None  # Keeps the last SendMessageAction

    @property
    def evaluation_fn(self) -> Callable[[List[str], list[str], List[Any]], tuple[int, int, List[str]]]:
        return self.evaluate

    def __post_init__(self):
        super().__post_init__()
        assert self.cost_callback is not None, "Dynamic examples require a cost callback."
        assert self.max_score > 0

    def evaluate(self, *args, **kwargs) -> tuple[int, int, list[str]]:
        return self.score, self.max_score, self.reasoning

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

    def wait(self, tokens: Optional[int] = None, time: Optional[timedelta] = None) -> WaitAction:
        kwargs = {k: v for k, v in dict(tokens=tokens, time=time).items() if v is not None}
        if tokens is None and time is None:
            kwargs["tokens"] = randint(self.filler_tokens_low, self.filler_tokens_high)
        return WaitAction(**kwargs)

    def say(self, message: str, question: bool = False) -> SendMessageAction:
        # `question = True` will register the answer as `result.actual_response`
        self.action = SendMessageAction(message=message, is_question=question)
        self.script.append(message)
        return self.action


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

    @property
    def data_path(self) -> Path:
        return DATA_DIR.joinpath(self.name)

    def load_file(self, name: str) -> str:
        with open(self.data_path.joinpath(name)) as fd:
            return fd.read()

    def load_json(self, name: str, **kwargs) -> Any:
        return json.loads(self.load_file(name), **kwargs)

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

    def default_waits(self, is_question: list[bool], current_waits: list[dict[str, Any]] = []):
        def _filler_size(is_p2q: bool):
            return self.pre_question_filler if is_p2q else (randint(self.filler_tokens_low, self.filler_tokens_high))

        assert len(is_question) >= 1, "There are no questions for this test"
        assert len(current_waits) == 0 or len(current_waits) == len(is_question), "Current waits should be empty or the same length as the script"

        # Create the empty waits if there are any.
        if len(current_waits) == 0:
            for _ in range(len(is_question)):
                current_waits.append({})

        is_prior_to_question = is_question[1:] + [False]
        for idx, (f, p2q) in enumerate(zip(current_waits[:-1], is_prior_to_question)):
            if f != {}:
                continue

            current_waits[idx] = WaitCreator.create_wait(tokens=_filler_size(p2q))

        current_waits[-1] = WaitCreator.create_wait()
        return current_waits

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
            filler_tokens_low=self.filler_tokens_low,
            filler_tokens_high=self.filler_tokens_high,
            **kwargs
        )

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        return [self.create_example() for _ in range(num_examples)]
