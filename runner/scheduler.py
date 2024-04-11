import json
import math
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterator, List, Tuple, Callable

import time_machine

from dataset_interfaces.interface import (
    TestExample,
    DynamicExample,
    SendMessageAction,
    WaitAction,
    SendAndRegisterAction,
)
from model_interfaces.interface import ChatSession
from reporting.generate import generate_report
from reporting.results import TestResult
from runner.config import RunConfig
from runner.master_log import MasterLog
from utils.constants import EventType
from utils.filling_task import filler_no_response_tokens_trivia
from utils.ui import colour_print
from utils.files import make_runstats_path, make_master_log_path
from runner.progress import ProgressDialog


def are_compatible(a: TestExample, b: TestExample, incompatibilities: list[set[type]]) -> bool:
    cls_a, cls_b = type(a.dataset_generator), type(b.dataset_generator)
    if cls_a is cls_b:
        return False
    for inc_set in incompatibilities:
        if cls_a in inc_set and cls_b in inc_set:
            return False
    return True


def create_question(example: TestExample, master_log: MasterLog) -> str:
    statement_times = [e.timestamp for e in master_log.test_events(example.unique_id, event_type=EventType.SEND_MESSAGE)]
    return example.dataset_generator.create_question(
        example,
        statement_times,
        datetime.now(),
    )


@dataclass
class TestRunner:
    config: RunConfig
    agent: ChatSession
    tests: list[TestExample]
    finished_results: list[TestResult] = field(default_factory=list)
    in_progress_results: dict[str, TestResult] = field(default_factory=dict)
    traveller: Optional[time_machine.travel] = None
    wait_list: dict[str, dict[str, int | datetime]] = field(default_factory=dict)
    agent_token_count: int = 0
    total_token_count: int = 0
    test_managing_costs_usd: float = 0
    agent_benchmark_duration: float = 0
    skip_evaluations: bool = False
    result_callbacks: List[Tuple[Callable, TestExample]] = field(default_factory=list)
    master_log: MasterLog = None
    progress_dialog: ProgressDialog = None

    @property
    def runstats_path(self):
        return make_runstats_path(self.config.run_name, self.agent.name)

    @property
    def master_log_path(self):
        return make_master_log_path(self.config.run_name, self.agent.name)

    @property
    def avg_tokens_per_second(self) -> float:
        return self.agent_token_count / (self.agent_benchmark_duration + 1e-8)

    def save_runstats(self):
        stats = dict(
            total_tokens=self.total_token_count,
            agent_tokens=self.agent_token_count,
            agent_costs_usd=self.agent.costs_usd,
            managing_costs_usd=self.test_managing_costs_usd,
            duration=self.agent_benchmark_duration,
        )
        with open(self.runstats_path, "w") as fd:
            json.dump(stats, fd)

    def load(self):
        if not self.runstats_path.exists():
            return
        assert (
                self.agent_benchmark_duration == 0
        ), "Attempted to load test scheduler in a non-initial state."
        with open(self.runstats_path) as fd:
            d = json.load(fd)
        self.agent_token_count = d["agent_tokens"]
        self.total_token_count = d["total_tokens"]
        self.agent.costs_usd = d["agent_costs_usd"]
        self.test_managing_costs_usd = d["managing_costs_usd"]
        self.agent_benchmark_duration = d["duration"]
        self.master_log.load()

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

    def reset_time(self):
        if self.traveller is not None:
            self.traveller.stop()
            self.traveller = None

    def set_to_wait(self, example: TestExample, action: WaitAction, log_this: bool = True):
        unique_id = example.unique_id

        # We convert the percentage to tokens
        if action.percentage_finished > 0.0:
            span_percent = example.dataset_generator.memory_span / 100
            percent_as_tokens = math.floor(span_percent * action.percentage_finished)

            # Compute at what token we are at relative to this test's memory span, then calculate the difference.
            tokens_elapsed = self.total_token_count - example.start_token
            token_wait = max(0, percent_as_tokens - tokens_elapsed, action.tokens)

        else:
            token_wait = action.tokens

        if log_this:
            self.master_log.add_wait_event(
                unique_id,
                datetime.now(),
                tokens=token_wait,
                time=action.time,
            )
        self.wait_list[unique_id] = dict(
            tokens=self.total_token_count + token_wait,
            time=datetime.now() + action.time,
        )

    def send_message(self, test_id: str, action: SendMessageAction) -> int:
        action.reply, action.sent_ts, action.reply_ts = self.agent.message_to_agent(action.message)
        self.debug_message(action.message, action.reply, action.sent_ts, action.reply_ts)
        self.master_log.add_send_message(
            test_id=test_id, message=action.message, timestamp=action.sent_ts, is_question=action.is_question,
        )
        self.master_log.add_response_message(
            test_id=test_id, message=action.reply, timestamp=action.reply_ts,  is_question=action.is_question
        )
        self.agent_benchmark_duration += (action.reply_ts - action.sent_ts).total_seconds()
        message_tokens = self.agent.token_len(action.message)
        reply_tokens = self.agent.token_len(action.reply)
        self.agent_token_count += reply_tokens
        self.total_token_count += message_tokens + reply_tokens
        return message_tokens + reply_tokens

    def get_blocked_test(self, waiting_on: str) -> Optional[str]:
        assert waiting_on in ["tokens", "time"]
        target = dict(
            tokens=self.total_token_count,
            time=datetime.now(),
        )[waiting_on]
        waiting_tests = {uid: wd for uid, wd in self.wait_list.items() if wd[waiting_on] > target}
        if len(waiting_tests) == 0:
            return
        return sorted(waiting_tests.keys(), key=lambda uid: waiting_tests[uid][waiting_on])[0]

    def is_waiting(self, unique_id: str, remove: bool = False) -> bool:
        if unique_id not in self.wait_list:
            return False
        wait_dict = self.wait_list[unique_id]
        if wait_dict["tokens"] > self.total_token_count:
            return True
        if wait_dict["time"] > datetime.now():
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

        # See first if any waiting test has met its waiting conditions in the meantime.
        if len(self.wait_list) > 0:
            waiting_ids = list(self.wait_list.keys())
            waiting_ids.sort(key=lambda i: self.wait_list[i].get("tokens", float("inf")))
            for unique_id in waiting_ids:
                if not self.is_waiting(unique_id, remove=True):
                    return unique_id

        # Otherwise, pick any other compatible test and start it.
        for unique_id in sorted(tests.keys()):
            if not self.is_compatible(tests[unique_id], tests):
                continue
            if not self.is_waiting(unique_id, remove=True):
                return unique_id

        # If all tests are waiting, try to meet the waiting conditions
        # Fast-forwarding time is easier than filling tokens, so do that first.
        while True:
            time_waiting_id = self.get_blocked_test("time")
            if time_waiting_id is None:
                break
            target_dt = self.wait_list[time_waiting_id]["time"]
            remaining_time = target_dt - datetime.now()
            waiting_time = max(remaining_time.total_seconds(), 0)
            self.forward_time(seconds=waiting_time + 1)
            if not self.is_waiting(time_waiting_id, remove=True):
                return time_waiting_id

        # Perform some filling task for token waiting
        while True:
            token_waiting_id = self.get_blocked_test("tokens")
            if token_waiting_id is None:
                break
            num_tokens = self.wait_list[token_waiting_id]["tokens"]
            remaining_tokens = num_tokens - self.total_token_count
            while remaining_tokens > 0:
                msg = filler_no_response_tokens_trivia(remaining_tokens, self.agent.max_message_size, self.agent)
                tokens_spent = self.send_message("", SendMessageAction(msg, is_filling=True))
                remaining_tokens -= tokens_spent
            if not self.is_waiting(token_waiting_id, remove=True):
                return token_waiting_id

        assert False, f"Couldn't find a test to run. Wait list: {self.wait_list}"

    def fast_forward_tests(self, tests: dict[str, TestExample]):
        # Go event by event in the master log and sync up with trace
        finished_tests = {evt.test_id for evt in self.master_log.log if evt.type == EventType.END}
        action = None
        for evt in self.master_log.log:
            if evt.type not in {EventType.BEGIN, EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE, EventType.WAIT}:
                continue
            if evt.test_id in finished_tests:
                continue
            test = tests[evt.test_id]
            self.wait_list.pop(evt.test_id, None)  # Since the test is performing an action, it can't be waiting.
            match evt.type:
                case EventType.BEGIN:
                    result, skip = self.initialise_result(test)
                    assert not skip
                    test.start_token = self.master_log.get_start_token(evt.test_id)
                    self.in_progress_results[evt.test_id] = result
                case EventType.SEND_MESSAGE:
                    action = test.step()
                    if action is None:
                        assert evt.data["message"] == test.reset_message
                    else:
                        assert isinstance(action, SendMessageAction)
                        assert action.message == evt.data["message"]
                        assert action.is_question == evt.data["is_question"]
                        assert not action.is_filling
                        action.sent_ts = evt.timestamp
                        if isinstance(action, SendAndRegisterAction):
                            self.register_callback(test)
                case EventType.RESPONSE_MESSAGE:
                    if action is not None:
                        assert isinstance(action, SendMessageAction)
                        assert action.is_question == evt.data["is_question"]
                        action.reply = evt.data["message"]
                        action.reply_ts = evt.timestamp
                case EventType.WAIT:
                    action = test.step()
                    assert isinstance(action, WaitAction)
                    assert action.tokens == evt.data["tokens"] or action.percentage_finished > 0.0
                    assert action.time == evt.data["time"]
                    self.travel_to_dt(evt.timestamp)
                    self.set_to_wait(test, action, log_this=False)
                    self.reset_time()

    def setup_iterator(self, test_group):
        """Sets up the test dict and fast forwards any tests that are currently in progress"""
        tests = {t.unique_id: t for t in test_group}

        # Give dynamic tests access to the master log for LLM caching
        for t in tests.values():
            if isinstance(t, DynamicExample):
                t.master_log = self.master_log

        # Fast forward tests to where they were
        self.fast_forward_tests(tests)

        # Check if the last event in the log is after the current time, then travel to that time if it is.
        if len(self.master_log.log) > 0 and datetime.now() < self.master_log.log[-1].timestamp:
            self.travel_to_dt(self.master_log.log[-1].timestamp)

        # Add a reset event to the log if it has indeed been reset
        if len(self.master_log.log) > 0:
            self.master_log.add_reset_event(datetime.now())

        return tests

    def iter_tests(self, test_group: list[TestExample]) -> Iterator[TestExample]:
        # Set up the tests that are to be iterated through, fast forwarding where appropriate
        tests = self.setup_iterator(test_group)

        assert len(tests) == len(test_group), f"There are tests with identical IDs: {[t.unique_id for t in test_group]}"
        while len(tests) > 0:
            run_id = self.pick_next_test_id(tests)
            example = tests[run_id]
            yield example
            if example.finished:
                del tests[run_id]
                self.wait_list.pop(run_id, None)

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
        skip = result.path.exists()
        if skip:
            result.load()
        return result, skip

    def run_tests(self):
        self.result_callbacks = []
        self.in_progress_results = dict()
        finished = 0

        # Introduce the benchmark, if running from the start.
        if len(self.master_log.log) == 0:
            self.send_message("", SendMessageAction(message=(
                "I am going to subject you to a Long-Term Memory benchmark. In the following, I will be giving you "
                "different kinds of information and I expect you to answer extremely briefly, only providing the "
                "responses that you are required to provide. Otherwise, provide just short confirmations. Understood?"
            )))

        for example in self.iter_tests(self.tests):
            self.progress_dialog.notify_running(example)

            skip = False
            if example.unique_id not in self.in_progress_results:
                result, skip = self.initialise_result(example)
                self.in_progress_results[example.unique_id] = result
                example.finished = skip
                if not skip:
                    self.master_log.begin_test(example.unique_id, datetime.now(), self.total_token_count)
                    if example.start_token == 0:
                        example.start_token = self.total_token_count

            while not example.finished:
                action = example.step()
                if action is None:
                    break
                if isinstance(action, WaitAction):
                    self.set_to_wait(example, action)
                    break
                if isinstance(action, (SendMessageAction, SendAndRegisterAction)):
                    # TODO: the test should autonomously create the question
                    if example.is_temporal and action.is_question:
                        action.message = create_question(example, self.master_log)
                    self.send_message(example.unique_id, action)
                    if isinstance(action, SendAndRegisterAction):
                        self.register_callback(example)

                self.save_runstats()
                self.agent.save()
            self.check_result_callbacks()

            if example.finished:
                finished += 1
                result = self.in_progress_results[example.unique_id]
                self.progress_dialog.notify_result(result)

                if not skip:
                    if example.reset_message != "":
                        action = SendMessageAction(example.reset_message)
                        self.send_message(example.unique_id, action)

                    self.update_result(
                        example=example,
                        result=result,
                        master_log=self.master_log,
                    )
                    self.master_log.end_test(example.unique_id, datetime.now())
                    self.agent.save()

                self.finished_results.append(result)
                print(result)
                colour_print("green", f"{finished} of {len(self.tests)} tests finished.")

    def register_callback(self, example: TestExample):
        cb = example.dataset_generator.continual_evaluation_callback
        self.result_callbacks.append((cb, example))

    def check_result_callbacks(self):
        deregistered_cb = []
        for callback, example in self.result_callbacks:
            score, max_score, reasons, deregister = callback(self, example, self.master_log.messages())
            result = self.in_progress_results[example.unique_id]
            result.score = score
            result.max_score = max_score
            result.reasoning = reasons

            self.update_result(example, result, self.master_log)

            if deregister:
                assert example.finished, "Callback has been deregistered, but example is not set as finished!"
                deregistered_cb.append((callback, example))

        for tup in deregistered_cb:
            self.result_callbacks.remove(tup)

    def set_cost_callback(self):
        def cost_callback(cost_usd: float):
            self.test_managing_costs_usd += cost_usd

        for example in self.tests:
            example.dataset_generator.cost_callback = cost_callback

    def run(self):
        self.master_log = MasterLog(self.master_log_path)
        self.runstats_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_path.mkdir(parents=True, exist_ok=True)
        self.load()
        self.set_cost_callback()
        colour_print("green", f"Number of tests to run: {len(self.tests)}.")
        self.tests.sort(key=lambda t: t.unique_id)
        self.progress_dialog = ProgressDialog(len(self.tests))
        self.run_tests()
        self.progress_dialog.close()
        self.save_runstats()
        self.reset_time()
        report_path = generate_report(self.finished_results)
        webbrowser.open_new_tab(report_path.as_uri())

    def update_result(
        self,
        example: TestExample,
        result: TestResult,
        master_log: MasterLog,
    ):
        task_log = master_log.messages(example.unique_id)
        questions, question_responses = master_log.get_questions_and_responses(example.unique_id)

        characters, tokens = example.dataset_generator.tokens_to_answer(
            master_log.as_context(example.unique_id),
            master_log.as_context(),
            example
        )

        if tokens > example.dataset_generator.memory_span:
            colour_print("RED", f"WARN: This test passed the memory_span threshold. The threshold was {example.dataset_generator.memory_span} tokens, while the memory span of the test was {tokens} tokens."
                  " If you are relying on the test being inside of the memory span, then any test failures could be caused by this overrun.")

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

        result.needles = len(example.script) - example.number_of_questions
        result.task_log = task_log
        result.actual_responses = question_responses
        result.tokens = tokens
        result.full_log = self.master_log.human_readable_full_log(example.unique_id, example.script[0])
        result.characters = characters
        result.save()
        self.save_runstats()

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
