from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import json
from utils.files import make_result_path, parse_result_path
from dataset_interfaces.factory import DATASETS_BY_NAME


@dataclass
class TestResult:
    run_name: str
    agent_name: str
    dataset_name: str
    example_id: str
    description: str = ""
    task_log: List[str] = field(default_factory=list)
    expected_responses: List[str] = field(default_factory=list)
    actual_responses: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    score: int = 0
    max_score: int = 0
    tokens: int = None
    characters: int = None
    repetition: int = 0
    full_log: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._saved_attrs = [
            "task_log",
            "actual_responses",
            "score",
            "max_score",
            "reasoning",
            "tokens",
            "characters",
            "full_log",
            "expected_responses",
        ]

    def __str__(self):
        string = ""
        string += f"Dataset Name: {self.dataset_name}\n"
        string += f"Run Name: {self.run_name}\n"
        string += f"Description: {self.description}\nTask log:\n"
        for idx, s in enumerate(self.task_log):
            string += f"\t{s}\n"
            if (idx + 1) % 2 == 0:
                string += "\n"
        string += f"\nExpected response: {self.expected_responses}\n"
        string += f"\nActual response: {' '.join(self.actual_responses)}\n"
        string += f"\nReasoning: {self.reasoning}\n"
        string += f"\nScore: {self.score}/{self.max_score}\n"
        string += f"Tokens: {self.tokens}\n"
        string += f"Characters: {self.characters}\n"

        return string

    def path(self, agent: str) -> Path:
        return make_result_path(self.run_name, agent, self.dataset_name, self.example_id, self.repetition)

    def save(self, agent: str):
        file_path = self.path(agent)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fd:
            json.dump({k: getattr(self, k) for k in self._saved_attrs}, fd, indent=2)

    def load(self, agent: str):
        with open(self.path(agent)) as fd:
            d = json.load(fd)
        for k in self._saved_attrs:
            setattr(self, k, d[k])

    @classmethod
    def from_file(cls, path: Path | str) -> "TestResult":
        result = TestResult(**parse_result_path(path))
        result.description = DATASETS_BY_NAME[result.dataset_name].description
        result.load(result.agent_name)
        return result
