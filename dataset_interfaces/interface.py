import json
import math
from copy import deepcopy
from random import Random
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Callable, Tuple, Optional, Any, Iterator, Dict
from runner.master_log import MasterLog

import tiktoken
from goodai.helpers.json_helper import sanitize_and_parse_json

from utils.constants import DATA_DIR
from utils.context import flatten_context, search_context
from utils.llm import ask_llm, LLMContext, get_tokens_for_script
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
""".strip()


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


class WaitCreator:
    @classmethod
    def create_wait(cls, tokens: int = None, time: timedelta = None):
        w_dict = {"tokens": tokens, "time": time}
        return {k: v for k, v in w_dict.items() if v is not None}

    @classmethod
    def unserialise(cls, w_dict):
        w_dict = deepcopy(w_dict)
        if "time" in w_dict:
            w_dict["time"] = timedelta(seconds=w_dict["time"])
        return w_dict

    @classmethod
    def serialise(cls, w_dict):
        w_dict = deepcopy(w_dict)
        if "time" in w_dict:
            w_dict["time"] = w_dict["time"].seconds
        return w_dict


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
    memory_span: int = None
    random: Random = None  # Seeded random generator

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
        assert self.memory_span is not None
        self.number_of_questions = len([q for q in self.is_question if q])
        self._iter = self.action_iter()
        self.default_waits()
        self.random = Random(self.unique_id)

    def action_iter(self) -> Iterator[TestAction]:
        scripts = [self.script, self.waits, self.is_question]
        for msg, wait, is_q in zip(*scripts):
            if self.uses_callback and is_q:
                yield SendAndRegisterAction(msg, is_question=is_q)
            else:
                yield SendMessageAction(msg, is_question=is_q)
            if len(wait) > 0:
                yield WaitAction(**wait)
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
            memory_span=self.memory_span,
        )

    def save(self, run_name: str, exist_ok: bool = False):
        file_path = self.get_path(run_name)
        if not exist_ok:
            assert not file_path.exists(), f"Attempt to overwrite test {file_path}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fd:
            json.dump(self.to_dict(), fd, indent=2)

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

    def default_waits(self):

        if isinstance(self, DynamicExample):
            return

        assert len(self.is_question) >= 1, "There are no questions for this test"
        assert len(self.waits) == 0 or len(self.waits) == len(self.is_question), "Current waits should be empty or the same length as the script"

        # TODO: All this token counting is based on OpenAIs methods, which is not Anthropics - but Aanthopic has no reliable tokeniser released so....
        script_tokens = get_tokens_for_script(self.script)
        assert script_tokens < self.memory_span, f"The script for test {self.dataset_name} is too long (estimated {script_tokens} tokens) for the specified memory span of {self.memory_span} tokens."

        span_minus_script = self.memory_span - script_tokens

        # Figure out what our spacing is going to be. We will do the 75%/25% needle distribution.
        tokens_for_needles = math.floor(span_minus_script * .75)
        needles = len(self.is_question) - sum(self.is_question)
        token_wait_per_needle = tokens_for_needles // needles
        token_wait_between_questions = math.floor(span_minus_script * .9 * .25) // sum(self.is_question)

        # Create the empty waits if there are any.
        if len(self.waits) == 0:
            for _ in range(len(self.is_question)):
                self.waits.append({})

        total_wait_tokens = 0

        for idx, f in enumerate(self.waits[:-1]):
            if f != {}:
                continue

            wait = token_wait_per_needle if idx < needles else token_wait_between_questions
            self.waits[idx] = WaitCreator.create_wait(tokens=int(wait))
            total_wait_tokens += wait

        assert total_wait_tokens < span_minus_script, "Sum of task waits is higher than the determined memory span."
        self.waits[-1] = WaitCreator.create_wait()


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
    score: int = 0
    max_score: int = 0
    expected_responses: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    script: List[str] = field(default_factory=lambda: [])  # Updated dynamically by `say` method
    action: SendMessageAction = None  # Keeps the last SendMessageAction
    wait = WaitAction
    llm_call_idx: int = -1
    master_log: MasterLog = None  # Set by runner to cache llm calls

    @property
    def evaluation_fn(self) -> Callable[[List[str], list[str], List[Any]], tuple[int, int, List[str]]]:
        return self.evaluate

    def __post_init__(self):
        super().__post_init__()
        assert self.max_score > 0

    def evaluate(self, *args, **kwargs) -> tuple[int, int, list[str]]:
        return self.score, self.max_score, self.reasoning

    def ask_llm(self, context: LLMContext, **kwargs) -> str:
        self.llm_call_idx += 1
        response = self.master_log.get_cached_response(self.unique_id, self.llm_call_idx)
        if response is not None:
            return response
        response = self.dataset_generator.ask_llm(context, **kwargs)
        self.master_log.add_llm_call(self.unique_id, datetime.now(), response)
        return response

    @abstractmethod
    def action_iter(self) -> Iterator[TestAction]:
        pass

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
    memory_span: int = 0
    seed: int = 0
    cost_callback: Callable[[float], None] = None
    uses_callback: bool = False
    reset_message: str = ""
    max_message_size: int = 1024
    random: Random = None  # Seeded random generator

    def __getattribute__(self, item):
        # Force seeding right before generating samples
        if item == "generate_examples":
            self.random = Random(self.seed)
        return super().__getattribute__(item)

    @property
    def data_path(self) -> Path:
        return DATA_DIR.joinpath(self.name)

    def load_file(self, path_or_name: str | Path) -> str:
        path = Path(path_or_name)
        if not path.is_absolute():
            path = self.data_path.joinpath(path_or_name)
        with open(path) as fd:
            return fd.read()

    def load_json(self, path_or_name: str | Path, **kwargs) -> Any:
        return json.loads(self.load_file(path_or_name), **kwargs)

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

    def ask_llm(self, context: LLMContext, temperature: float = 0, max_tokens: int = 256, **kwargs) -> str:
        return ask_llm(
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            cost_callback=self.cost_callback,
            **kwargs,
        )

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

        response = ask_llm(context=ctx, model="gpt-4-0125-preview", temperature=0.01, cost_callback=cost_callback)
        score = 0
        reasoning = []
        try:
            parsed = sanitize_and_parse_json(response)

            for evaluation in parsed:
                yes_no_list = [yn.lower() == "yes" for yn in evaluation["checklist"]]

                correct = yes_no_list[0]  # reference
                if yes_no_list[1] and not yes_no_list[2]:
                    correct = False
                score += int(correct)

                mk_ref_str = "makes" if yes_no_list[0] else "does not make"
                is_num_str = "is" if yes_no_list[1] else "is not"
                reason = f"The answer {mk_ref_str} reference to the expected answer, which {is_num_str} numerical."
                if yes_no_list[1]:
                    match_str = "match" if yes_no_list[2] else "do not match"
                    reason = reason[:-1] + f", and the numbers {match_str}."
                reasoning.append(reason)
        except:
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
        if self.memory_span > 0:
            kwargs["memory_span"] = self.memory_span
        return self.example_cls(dataset_generator=self, **kwargs)

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        return [self.create_example() for _ in range(num_examples)]

    def default_waits(self, is_question: list[bool], current_waits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return []
