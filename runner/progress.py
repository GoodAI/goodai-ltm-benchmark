import tkinter as tk
from tkinter import ttk
from dataset_interfaces.interface import TestExample
from reporting.results import TestResult
from collections import defaultdict
from utils.math import mean_std


def blinker_gen():
    while True:
        for c in r"/-\|":
            yield c


class ProgressDialog(tk.Tk):
    def __init__(self, num_tests: int, isolated: bool):
        super().__init__()
        self._num_tests = num_tests
        self._isolated = isolated
        self._memory_span = None
        self._at = 0
        self._test_info = dict()
        self._scores = defaultdict(lambda: list())
        self._blinker = blinker_gen()

        self.title("GoodAI LTM Benchmark")

        self._label = tk.Label(self, text="Setting up...")
        self._label.pack(padx=20, pady=(20, 5))

        self._progressbar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self._progressbar.pack(padx=20, pady=5)
        self.update_idletasks()

    def notify_running(self, example: TestExample):
        self._at = max(self._at, example.start_token)
        self._memory_span = self._memory_span or example.dataset_generator.memory_span
        self._test_info[example.unique_id] = dict(start=example.start_token, span=self._memory_span)
        self.update_stats()

    def notify_message(self, token_count: int):
        self._at = token_count
        self.update_stats()

    def notify_result(self, result: TestResult):
        info = self._test_info[result.unique_id]
        info["span"] = self._at - info["start"]
        self._scores[result.dataset_name].append(result.score / result.max_score)
        self.update_stats()

    def update_stats(self):
        if len(self._test_info) == 0:
            return

        total_score = total_std = 0
        for scores in self._scores.values():
            score, std = mean_std(scores)
            total_score += score
            total_std += std
        self._label.config(text=f"{next(self._blinker)} Score: {total_score:.1f} ± {total_std:.1f}")

        if self._isolated:
            progress = len(self._test_info) / self._num_tests
        else:
            total = [info["span"] for info in self._test_info.values()]
            total += [self._memory_span] * (self._num_tests - len(total))
            progress = sum(min(max(0, self._at - info["start"]), info["span"]) for info in self._test_info.values())
            progress /= max(sum(total), 1)
        self._progressbar["value"] = int(100 * progress)
        self.update_idletasks()

    def close(self):
        self.destroy()
