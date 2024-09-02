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
from utils.llm import ask_llm, LLMContext, count_tokens_for_model
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


def normalize_scores(evaluate_correct_fn: Callable[[List[str], List[str], List[Any]], Tuple[int, int, List[str]]],
                     questions: List[str], responses: List[str], expected_answers: List[Any]) -> Tuple[float, int, List[str]]:
    correct, total, feedback = evaluate_correct_fn(questions, responses, expected_answers)
    normalized_score = float(correct) / total if total > 0 else 0.0
    return normalized_score, 1, feedback


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
    filler_response: str = None


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
    def create_wait(cls, tokens: int = None, time: timedelta = None, percentage_finished: float = None):
        w_dict = {"tokens": tokens, "time": time, "percentage_finished": percentage_finished}
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
    random: Random = None  # Seeded random generator
    start_token: int = 0

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
    def evaluation_fn(self) -> Callable[[List[str], List[str], List[Any]], Tuple[float, int, List[str]]]:
        """
        Returns a callable that evaluates and normalizes the scores between 0 and 1.
        The returned tuple consists of the normalized score, the max score (always 1 for normalized scores), and feedback.
        """
        def evaluator(questions: List[str], responses: List[str], expected_answers: List[Any]) -> Tuple[float, int, List[str]]:
            return normalize_scores(self.dataset_generator.evaluate_correct, questions, responses, expected_answers)
        return evaluator

    @property
    def unique_id(self):
        return f"{self.dataset_name} - {self.example_id}"

    def get_path(self, run_name: str) -> Path:
        return make_testdef_path(run_name, self.dataset_name, self.example_id)

    def __post_init__(self):
        assert self.dataset_generator is not None
        assert self.dataset_generator.memory_span > 0
        self.number_of_questions = len([q for q in self.is_question if q])
        self._iter = self.action_iter()
        self.random = Random(self.unique_id)
        self.default_waits()

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
        )

    def save(self, run_name: str, exist_ok: bool = False):
        file_path = self.get_path(run_name)
        if not exist_ok:
            assert not file_path.exists(), f"Attempt to overwrite test {file_path}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fd:
            json.dump(self.to_dict(), fd, indent=2)

    @classmethod
    def load(cls, dataset_generator: "DatasetInterface", file_path: Path | str) -> "TestExample":

        file_path = Path(file_path)
        assert file_path.exists(), f"Test file doesn't exist: {file_path}"
        assert file_path.name.endswith(".def.json")
        with open(file_path) as fd:
            d = json.load(fd)
        d["waits"] = [WaitCreator.unserialise(x) for x in d["waits"]]
        d["example_id"] = file_path.name.removesuffix(".def.json")
        d["dataset_name"] = file_path.parent.name
        generator_attrs = {"dataset_name", "description", "reset_message"}
        d = {k: d[k] for k in d.keys() if k not in generator_attrs}
        return dataset_generator.create_example(**d)

    def default_waits(self):

        if isinstance(self, DynamicExample):
            return

        # For caution, we want to use a tokeniser that produces the most tokens. This is to try and avoid the problem of
        # having the test overrun its memory_span. More tokens in script means smaller token gaps and more buffer.
        memory_span = self.dataset_generator.memory_span
        script_tokens = count_tokens_for_model(script=self.script)
        assert script_tokens < memory_span, (
            f"The script for test {self.dataset_name} is too long (estimated {script_tokens} tokens) for the specified "
            f"memory span of {memory_span} tokens."
        )

        assert sum(self.is_question) >= 1, "There are no questions for this test"
        num_script_lines = len(self.is_question)  # self.script might not be updated just yet [e.g. create_question()]
        if len(self.waits) == num_script_lines:
            return  # Do not overwrite waits
        assert len(self.waits) == 0, "Current waits should be empty in order to apply the defaults."

        perc_fraction = 90 / (num_script_lines - 1)
        self.waits = [
            WaitCreator.create_wait(
                percentage_finished=(i + 1) * perc_fraction,
            ) for i in range(num_script_lines - 1)
        ]
        self.waits.append(WaitCreator.create_wait())



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
    def evaluation_fn(self) -> Callable[[List[str], list[str], List[Any]], tuple[float, int, List[str]]]:
        return self.evaluate

    def __post_init__(self):
        super().__post_init__()
        assert self.max_score > 0

    def evaluate(self, *args, **kwargs) -> tuple[float, int, list[str]]:
        return self.score / self.max_score, 1, self.reasoning

    def ask_llm(self, context: LLMContext, model: str, **kwargs) -> str:
        self.llm_call_idx += 1
        response = self.master_log.get_cached_response(self.unique_id, self.llm_call_idx)
        if response is not None:
            return response
        response = self.dataset_generator.ask_llm(context, model, **kwargs)
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


    @abstractmethod
    def generate_reference_data(self, example: TestExample) -> Dict[str, Any]:
       pass

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

    def evaluation_fn(self) -> Callable[[List[str], List[str], List[Any]], Tuple[float, int, List[str]]]:
        def evaluator(questions: List[str], responses: List[str], expected_answers: List[Any]) -> Tuple[float, int, List[str]]:
            return normalize_scores(self.evaluate_correct, questions, responses, expected_answers)
        return evaluator

    def ask_llm(self, context: LLMContext, model: str, temperature: float = 0, max_tokens: int = 256, **kwargs) -> str:
        return ask_llm(
            context=context,
            model=model,
            temperature=temperature,
            max_response_tokens=max_tokens,
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

        response = ask_llm(context=ctx, model="gpt-4-turbo", temperature=0.01, cost_callback=cost_callback)
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
            **kwargs,
        )

    def tokens_to_answer(self, test_context: List[Dict[str, Any]], full_context: List[Dict[str, str]], example: TestExample):
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = num_characters = 0

        # Get the first line of the script
        relevant_line = example.script[0]

        timestamp_idx = search_context(test_context, relevant_line)
        target_timestamp = test_context[timestamp_idx]["timestamp"].__str__()

        # Where in the history was the statement made?
        # Find that statement in the history suing the content and timestamp and count from there.
        history_idx = search_context(full_context, relevant_line, target_timestamp)
        countable_history_chunk = flatten_context(full_context[history_idx:])

        # Now count the tokens and characters since there
        num_characters += len(countable_history_chunk)
        num_tokens += len(encoding.encode(countable_history_chunk, disallowed_special=()))

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

    def create_example(self, **kwargs) -> DynamicExample:
        return self.example_cls(dataset_generator=self, **kwargs)

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        return [self.create_example() for _ in range(num_examples)]

    def default_waits(self, script: list[str], is_question: list[bool], current_waits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return []
