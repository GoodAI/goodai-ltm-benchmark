import tkinter as tk
from tkinter import ttk
from dataset_interfaces.interface import TestExample
from reporting.results import TestResult
from collections import defaultdict
from utils.math import mean_std
from datetime import timedelta


def blinker_gen():
    while True:
        for c in r"/-\|":
            yield c


class ProgressDialog(tk.Tk):
    def __init__(self, tests: list[TestExample], isolated: bool):
        super().__init__()
        self._num_tests = len(tests)
        self._isolated = isolated
        self._memory_span = None
        self._at = 0
        self._duration = 0.0
        self._cost = 0.0
        self._test_info = dict()
        self._scores = defaultdict(lambda: list())
        self._blinker = blinker_gen()
        self._repetitions = defaultdict(lambda: 0)
        for t in tests:
            self._repetitions[t.dataset_generator.name] += 1

        self.title("GoodAI LTM Benchmark")

        self._1st_row = tk.Label(self, text="Setting up...")
        self._1st_row.pack(padx=20, pady=(20, 5))
        self._2nd_row = tk.Label(self, text="")
        self._2nd_row.pack(padx=20, pady=5)

        self._progressbar = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self._progressbar.pack(padx=20, pady=5)
        self.update_idletasks()

    def notify_running(self, example: TestExample):
        self._at = max(self._at, example.start_token)
        self._memory_span = self._memory_span or example.dataset_generator.memory_span
        self._test_info[example.unique_id] = dict(start=example.start_token, span=self._memory_span)
        self.update_stats()

    def notify_message(self, token_count: int, duration: float, cost: float):
        self._at = token_count
        self._duration = duration
        self._cost = cost
        self.update_stats()

    def notify_result(self, result: TestResult):
        info = self._test_info[result.unique_id]
        info["span"] = self._at - info["start"]
        self._scores[result.dataset_name].append(result.score / result.max_score)
        self.update_stats()

    def update_stats(self):
        if len(self._test_info) == 0:
            return

        total_tokens = [info["span"] for info in self._test_info.values()]
        total_tokens += [self._memory_span] * (self._num_tests - len(total_tokens))
        total_tokens = sum(total_tokens)

        # Progress bar shows the overall progress
        if self._isolated:
            progress = len(self._test_info) / self._num_tests
        else:
            progress = sum(min(max(0, self._at - info["start"]), info["span"]) for info in self._test_info.values())
            progress /= max(total_tokens, 1)
        self._progressbar["value"] = int(100 * progress)

        # First row shows the total score so far and the tests completed
        num_finished = 0
        total_score = 0
        for dataset_name, scores in self._scores.items():
            for s in scores:
                total_score += s / self._repetitions[dataset_name]
                num_finished += 1
        test_counter = f"{num_finished}/{self._num_tests}"
        self._1st_row.config(text=f"{next(self._blinker)} Score: {total_score:.1f}   Finished: {test_counter}")

        # Second row shows estimated time to finish and cost.
        time_est = "unknown"
        cost_est = "unknown"
        if progress > 0 and self._at > 128:  # Arbitrary num. of tokens to start estimating
            if self._duration > 0:
                seconds_to_finish = (self._duration / progress) * (1 - progress)
                time_est = str(timedelta(seconds=seconds_to_finish)).split(".")[0]
            cost_est = f"${self._cost / progress:.2f}"

        self._2nd_row.config(text=f"Time left: {time_est}   Est. cost: {cost_est}")

        self.update_idletasks()

    def close(self):
        self.destroy()
