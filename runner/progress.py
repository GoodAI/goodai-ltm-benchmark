import tkinter as tk
from tkinter import ttk
from dataset_interfaces.interface import TestExample
from reporting.results import TestResult


class ProgressDialog(tk.Tk):
    def __init__(self, num_tests: int):
        super().__init__()
        self._num_tests = num_tests
        self._tests_run = 0

        self.title("GoodAI LTM Benchmark")

        self._label = tk.Label(self, text="Setting up...")
        self._label.pack(padx=20, pady=(20, 5))

        self._progressbar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self._progressbar.pack(padx=20, pady=5)
        self.update_idletasks()

    def notify_running(self, example: TestExample):
        self._label.config(text=f"Running {example.dataset_name} ({example.example_id})")
        self.update_idletasks()

    def notify_result(self, result: TestResult):
        self._tests_run += 1
        self._progressbar["value"] = (self._tests_run / self._num_tests) * 100
        accuracy = int(100 * result.score / max(result.max_score, 1))
        self._label.config(text=f"{result.dataset_name}: {result.score}/{result.max_score} ({accuracy}%)")
        self.update_idletasks()

    def close(self):
        self.destroy()
