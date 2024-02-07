import tkinter as tk
from tkinter import ttk
from dataset_interfaces.interface import TestExample
from reporting.results import TestResult


class ProgressDialog(tk.Tk):
    def __init__(self, test_groups: list[list[TestExample]]):
        super().__init__()
        self._group_sizes = [len(group) for group in test_groups]
        self._num_tests = sum(self._group_sizes)
        self._tests_run = 0
        self._group_i = 0
        self._group_test_i = 0

        self.title("GoodAI LTM Benchmark")

        self._label = tk.Label(self, text="Setting up...")
        self._label.pack(padx=20, pady=(20, 5))

        self._global_progressbar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self._global_progressbar.pack(padx=20, pady=5)

        self._group_progressbar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self._group_progressbar.pack(padx=20, pady=(5, 20))
        self.update_idletasks()

    def notify_result(self, result: TestResult):
        self._tests_run += 1
        self._group_test_i += 1
        if self._group_test_i == self._group_sizes[self._group_i]:
            self._group_test_i = 0
            self._group_i += 1
        self._group_i = min(self._group_i, len(self._group_sizes) - 1)
        self._global_progressbar["value"] = (self._tests_run / self._num_tests) * 100
        self._group_progressbar["value"] = (self._group_test_i / self._group_sizes[self._group_i]) * 100
        accuracy = int(100 * result.score / max(result.max_score, 1))
        self._label.config(text=f"{result.dataset_name}: {result.score}/{result.max_score} ({accuracy}%)")
        self.update_idletasks()

    def close(self):
        self.destroy()


# Example usage:
if __name__ == "__main__":
    import time
    from random import randint

    total_global_tasks = 100
    total_subtasks = 10

    fake_groups = [[0 for _ in range(i + 10)] for i in range(5)]
    progress_dialog = ProgressDialog(fake_groups)

    class Dummy:
        def __init__(self, dataset_name, score, max_score):
            self.dataset_name = dataset_name
            self.score = score
            self.max_score = max_score

    for group in fake_groups:
        for test in group:
            time.sleep(1)
            max_score = randint(1, 4)
            score = randint(0, max_score)
            result = Dummy(f"Test({test})", score, max_score)
            progress_dialog.notify_result(result)

    progress_dialog.close()
