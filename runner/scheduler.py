import json
import webbrowser
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterator, List, Tuple, Callable

import time_machine

from dataset_interfaces.interface import (
    TestExample,
    TestAction,
    SendMessageAction,
    WaitAction,
    SendAndRegisterAction,
)
from model_interfaces.interface import ChatSession
from reporting.generate import generate_report
from reporting.results import TestResult
from runner.config import RunConfig
from utils.filling_task import filler_no_response_tokens_trivia
from utils.tokens import token_len
from utils.ui import colour_print
from utils.files import make_runstats_path
from runner.progress import ProgressDialog


def are_compatible(a: TestExample, b: TestExample, incompatibilities: list[set[type]]) -> bool:
    cls_a, cls_b = type(a.dataset_generator), type(b.dataset_generator)
    if cls_a is cls_b:
        return False
    for inc_set in incompatibilities:
        if cls_a in inc_set and cls_b in inc_set:
            return False
    return True


def create_question(example: TestExample, action_logs: dict[str, list[TestAction]]) -> str:
    statement_times = []
    for action in action_logs[example.unique_id]:
        if not isinstance(action, SendMessageAction):
            continue
        statement_times.append(action.sent_ts)
    return example.dataset_generator.create_question(
        example,
        statement_times,
        datetime.now(),
    )


def extract_log_kwargs(action_log: list[TestAction]):
    kwargs = dict()
    for k in ["questions", "question_responses", "messages_timestamps", "task_log"]:
        kwargs[k] = []
    for action in action_log:
        if not isinstance(action, SendMessageAction):
            continue
        kwargs["task_log"].append(f"Test ({action.sent_ts}): " + action.message)
        kwargs["task_log"].append(f"Agent ({action.reply_ts}): " + action.reply)
        if not action.is_filling:
            kwargs["messages_timestamps"].append(action.sent_ts)
        if action.is_question:
            kwargs["questions"].append(action.message)
            kwargs["question_responses"].append(action.reply)
    return kwargs


@dataclass
class TestRunner:
    config: RunConfig
    agent: ChatSession
    tests: list[TestExample]
    results: list[TestResult] = field(default_factory=list)
    traveller: Optional[time_machine.travel] = None
    wait_list: dict[str, dict[str, int | datetime]] = field(default_factory=dict)
    current_token_count: int = 0
    avg_tokens_per_second: Optional[float] = None
    action_logs: dict[str, list[TestAction]] = field(default_factory=dict)
    test_managing_costs_usd: float = 0
    benchmark_duration_seconds: float = 0
    reference_duration_timestamp: Optional[datetime] = None
    skip_evaluations: bool = False
    result_callbacks: List[Tuple[Callable, TestExample, TestResult]] = field(default_factory=list)
    master_log: List[str] = field(default_factory=list)
    progress_dialog: ProgressDialog = None

    @property
    def saving_path(self):
        return make_runstats_path(self.config.run_name, self.agent.name)

    def save(self):
        self.update_duration()
        self.saving_path.parent.mkdir(parents=True, exist_ok=True)
        stats = dict(
            total_tokens=self.current_token_count,
            agent_costs_usd=self.agent.costs_usd,
            managing_costs_usd=self.test_managing_costs_usd,
            tokens_per_second=self.avg_tokens_per_second,
            duration=self.benchmark_duration_seconds,
        )
        with open(self.saving_path, "w") as fd:
            json.dump(stats, fd)

    def load(self):
        if not self.saving_path.exists():
            return
        assert (
            self.avg_tokens_per_second is None and self.reference_duration_timestamp is None
        ), "Attempted to load test scheduler in a non-initial state."
        with open(self.saving_path) as fd:
            d = json.load(fd)
        self.current_token_count = d["total_tokens"]
        self.agent.costs_usd = d["agent_costs_usd"]
        self.test_managing_costs_usd = d["managing_costs_usd"]
        self.avg_tokens_per_second = d["tokens_per_second"]
        self.benchmark_duration_seconds = d["duration"]

    def travel_to_dt(self, target_date: datetime):
        self.reset_time()
        self.traveller = time_machine.travel(target_date.astimezone(timezone.utc))
        self.traveller.start()

    def forward_time(self, **kwargs):
        t_jump = timedelta(**kwargs)
        if self.config.debug:
            colour_print("green", f"Time jump by {t_jump}")
        target_date = datetime.now() + t_jump
        assert target_date > datetime.now(), "Can only move forward in time. Going back is problematic."
        self.travel_to_dt(target_date)

    def update_duration(self):
        assert self.reference_duration_timestamp is not None
        dt = datetime.now()
        self.reset_time()
        current_dt = datetime.now()
        duration = (current_dt - self.reference_duration_timestamp).total_seconds()
        self.reference_duration_timestamp = current_dt
        self.benchmark_duration_seconds = (self.benchmark_duration_seconds or 0) + duration
        if dt > datetime.now():
            self.travel_to_dt(dt)

    def reset_time(self):
        if self.traveller is not None:
            self.traveller.stop()
            self.traveller = None

    def set_to_wait(self, unique_id: str, action: WaitAction):
        wait_dict = dict()
        if action.tokens > 0:
            wait_dict["tokens"] = self.current_token_count + action.tokens
        if action.time.total_seconds() > 0:
            wait_dict["time"] = datetime.now() + action.time
        self.wait_list[unique_id] = wait_dict

    def send_message(self, action: SendMessageAction):
        response, sent_ts, reply_ts = self.agent.message_to_agent(action.message)
        self.debug_message(action.message, response, sent_ts, reply_ts)
        self.master_log.append(action.message)
        self.master_log.append(response)
        response_time = (reply_ts - sent_ts).total_seconds()
        action.reply = response
        action.reply_ts = reply_ts
        action.sent_ts = sent_ts
        action_tokens = token_len(action.message) + token_len(action.reply)
        self.current_token_count += action_tokens
        tokens_per_second = action_tokens / (response_time + 1e-5)
        if self.avg_tokens_per_second is None:
            self.avg_tokens_per_second = tokens_per_second
        else:
            self.avg_tokens_per_second *= 0.9
            self.avg_tokens_per_second += 0.1 * tokens_per_second

        return action_tokens

    def get_waiting_test(self, waiting_on: str) -> Optional[str]:
        assert waiting_on in ["tokens", "time"]
        waiting_tests = dict()
        for unique_id, wait_dict in self.wait_list.items():
            if waiting_on not in wait_dict:
                continue
            waiting_ref = dict(
                time=datetime.now(),
                tokens=self.current_token_count,
            )[waiting_on]
            if wait_dict[waiting_on] > waiting_ref:
                waiting_tests[unique_id] = wait_dict
        if len(waiting_tests) == 0:
            return
        waiting_ids = list(waiting_tests.keys())
        waiting_ids.sort(key=lambda i: waiting_tests[i][waiting_on])
        return waiting_ids[0]

    def is_waiting(self, unique_id: str, remove: bool = False) -> bool:
        if unique_id not in self.wait_list:
            return False
        wait_dict = self.wait_list[unique_id]
        if "tokens" in wait_dict and wait_dict["tokens"] > self.current_token_count:
            return True
        if "time" in wait_dict and wait_dict["time"] > datetime.now():
            return True
        if remove:
            del self.wait_list[unique_id]
        return False

    def is_compatible(self, example: TestExample, tests: dict[str, TestExample]) -> bool:
        for waiting_id in self.wait_list.keys():
            if not are_compatible(example, tests[waiting_id], self.config.incompatibilities):
                return False
        return True

    def pick_next_test_id(self, tests: dict[str, TestExample]) -> str:

        # Prioritize waiting tests. Shorter waits go first.
        if len(self.wait_list) > 0:
            waiting_ids = list(self.wait_list.keys())
            waiting_ids.sort(key=lambda i: self.wait_list[i].get("tokens", float("inf")))
            for unique_id in waiting_ids:
                if not self.is_waiting(unique_id, remove=True):
                    return unique_id

        # Otherwise, pick any other compatible test
        for unique_id in sorted(tests.keys()):
            if not self.is_compatible(tests[unique_id], tests):
                continue
            if not self.is_waiting(unique_id, remove=True):
                return unique_id

        # If all tests are waiting, try to meet the waiting conditions
        # Fast-forwarding time is easier than filling tokens, so do that first.
        while True:
            time_waiting_id = self.get_waiting_test("time")
            if time_waiting_id is None:
                break
            target_dt = self.wait_list[time_waiting_id]["time"]
            remaining_time = target_dt - datetime.now()
            waiting_time = max(remaining_time.total_seconds(), 0)
            self.forward_time(seconds=waiting_time + 1)
            if not self.is_waiting(time_waiting_id, remove=True):
                return time_waiting_id

        # As a last resort, perform some filling task
        while True:
            token_waiting_id = self.get_waiting_test("tokens")
            if token_waiting_id is None:
                break
            num_tokens = self.wait_list[token_waiting_id]["tokens"]
            remaining_tokens = num_tokens - self.current_token_count
            while remaining_tokens > 0:
                msg = filler_no_response_tokens_trivia(remaining_tokens, self.agent.max_message_size)
                tokens_spent = self.send_message(SendMessageAction(msg, is_filling=True))
                remaining_tokens -= tokens_spent
            if not self.is_waiting(token_waiting_id, remove=True):
                return token_waiting_id

        assert False, f"Couldn't find a test to run. Wait list: {self.wait_list}"

    def iter_tests(self, test_group: list[TestExample]) -> Iterator[TestExample]:
        tests = {t.unique_id: t for t in test_group}
        assert len(tests) == len(test_group), f"There are tests with identical IDs: {[t.unique_id for t in test_group]}"
        while len(tests) > 0:
            run_id = self.pick_next_test_id(tests)
            example = tests[run_id]
            yield example
            if example.finished:
                del tests[run_id]
                self.wait_list.pop(run_id, None)

    def log_action(self, example: TestExample, action: TestAction):
        self.action_logs.setdefault(example.unique_id, []).append(action)

    def initialise_result(self, example: TestExample) -> tuple[TestResult, bool]:
        result = TestResult(
            example_id=example.example_id,
            agent_name=self.agent.name,
            run_name=self.config.run_name,
            dataset_name=example.dataset_name,
            description=example.description,
            expected_responses=example.expected_responses,
            max_score=1,
            score=1,
            reasoning=["Evaluation Skipped"],
        )
        skip = result.path(self.agent.name).exists()
        if skip:
            result.load(self.agent.name)
        return result, skip

    def run_tests(self):
        self.result_callbacks = []
        self.master_log = []
        results = dict()
        finished = 0
        for example in self.iter_tests(self.tests):
            self.progress_dialog.notify_running(example)

            skip = False
            if example.unique_id not in results:
                result, skip = self.initialise_result(example)
                results[example.unique_id] = result
                example.finished = skip

            while not example.finished:
                action = example.step()
                if action is None:
                    break
                self.log_action(example, action)  # Attributes are modified afterwards.
                if isinstance(action, WaitAction):
                    self.set_to_wait(example.unique_id, action)
                    break
                if isinstance(action, (SendMessageAction, SendAndRegisterAction)):
                    # TODO: the test should autonomously create the question
                    if example.is_temporal and action.is_question:
                        action.message = create_question(example, self.action_logs)
                    self.send_message(action)
                    if isinstance(action, SendAndRegisterAction):
                        self.register_callback(example, results[example.unique_id])

            self.check_result_callbacks()

            if example.finished:

                if not skip and example.reset_message != "":
                    action = SendMessageAction(example.reset_message)
                    self.log_action(example, action)
                    self.send_message(action)

                finished += 1
                result = results[example.unique_id]
                self.progress_dialog.notify_result(result)
                if not skip:
                    self.update_result(
                        example=example,
                        result=result,
                        **extract_log_kwargs(self.action_logs[example.unique_id]),
                    )
                self.results.append(result)
                print(result)
                colour_print("green", f"{finished} of {len(self.tests)} tests finished.")

    def register_callback(self, example: TestExample, result: TestResult):
        cb = example.dataset_generator.continual_evaluation_callback
        self.result_callbacks.append((cb, example, result))

    def check_result_callbacks(self):
        deregistered_cb = []
        for callback, example, result in self.result_callbacks:
            score, max_score, reasons, deregister = callback(self, example, self.master_log)
            result.score = score
            result.max_score = max_score
            result.reasoning = reasons

            self.update_result(example, result, **extract_log_kwargs(self.action_logs[example.unique_id]))

            if deregister:
                assert example.finished, "Callback has been deregistered, but example is not set as finished!"
                deregistered_cb.append((callback, example, result))

        for tup in deregistered_cb:
            self.result_callbacks.remove(tup)

    def set_cost_callback(self):
        def cost_callback(cost_usd: float):
            self.test_managing_costs_usd += cost_usd

        for example in self.tests:
            example.dataset_generator.cost_callback = cost_callback

    def run(self):
        self.load()
        self.reference_duration_timestamp = datetime.now()
        self.set_cost_callback()
        colour_print("green", f"Number of tests to run: {len(self.tests)}.")
        self.tests.sort(key=lambda t: t.unique_id)
        self.progress_dialog = ProgressDialog(len(self.tests))
        self.run_tests()
        self.progress_dialog.close()
        self.save()
        self.reset_time()
        report_path = generate_report(self.results)
        webbrowser.open_new_tab(report_path.as_uri())

    def update_result(
        self,
        example: TestExample,
        result: TestResult,
        questions: list[str],
        question_responses: list[str],
        messages_timestamps: list[datetime],
        task_log: list[str],
    ):
        characters, tokens = example.dataset_generator.tokens_to_answer(
            self.agent.history,
            example,
            messages_timestamps,
        )

        if not self.skip_evaluations:
            if not example.uses_callback:
                score, max_score, reason = example.evaluation_fn(
                    questions,
                    question_responses,
                    example.expected_responses,
                )

                result.score = score
                result.max_score = max_score
                result.reasoning = reason

        result.task_log = task_log
        result.actual_responses = question_responses
        result.tokens = tokens
        list_idx = self.master_log.index(example.script[0])
        result.full_log = deepcopy(self.master_log[list_idx:])
        result.characters = characters
        result.save(self.agent.name)
        self.save()

    def debug_message(
        self,
        user_message: str,
        response: str,
        user_timestamp: datetime,
        ai_timestamp: datetime,
    ):
        if not self.config.debug:
            return
        colour_print("cyan", f"{user_timestamp} User: {user_message}")
        colour_print("red", f"{ai_timestamp} Agent: {response}")
        print()
