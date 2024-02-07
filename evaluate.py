import re
import click
from datetime import datetime
from dataset_interfaces.interface import TestExample
from reporting.results import TestResult
from utils.files import gather_testdef_files, make_result_path
from utils.ui import colour_print, ask_yesno


def reconstruct_history(result: TestResult) -> list[dict[str, str | datetime]]:
    start_pattern = r"^((Test|Agent) (\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\))?:)"
    history = list()
    for line in result.task_log:
        m = re.match(start_pattern, line)
        history.append(
            dict(
                role="user" if m.group(2) == "Test" else "assistant",
                content=line.removeprefix(m.group(0) + " "),
                timestamp=datetime.fromisoformat(m.group(3).strip("()")),
            )
        )
    return history


def reconstruct_messages_timestamps(history: list[dict[str, str | datetime]], script: list[str]) -> list[datetime]:
    script_lines = set(script)
    return [msg["timestamp"] for msg in history if msg["content"] in script_lines]


def extract_questions(example: TestExample) -> list[str]:
    return [line for line, is_q in zip(example.script, example.is_question) if is_q]


@click.command("evaluate")
@click.argument("run_name", type=str)
@click.argument("agent_name", type=str)
@click.option("-y", required=False, is_flag=True, default=False, help="Automatically assent to questions")
def main(run_name: str, agent_name: str, y: bool):
    _main(run_name, agent_name, y)


def _main(run_name: str, agent_name: str, y: bool):
    examples = [TestExample.load(path) for path in gather_testdef_files(run_name)]
    results = list()
    for example in examples:
        result_path = make_result_path(run_name, agent_name, example.dataset_name, example.example_id, 0)
        assert result_path.exists(), f"Can't re-evaluate without an existing result file: {result_path}"
        colour_print("yellow", f"Evaluating {result_path}")
        result = TestResult.from_file(result_path)
        if not example.uses_callback:
            if example.is_temporal:
                # Get question from task log instead.
                questions = [result.task_log[-2].split(":")[1]]
            else:
                questions = extract_questions(example)

            result.score, result.max_score, result.reasoning = example.evaluation_fn(
                questions,
                result.actual_responses,
                example.expected_responses,
            )
        else:
            callback = example.dataset_generator.continual_evaluation_callback
            scheduler = None
            result.score, result.max_score, result.reasons, deregister = callback(scheduler, example, result.full_log)
            if not deregister:
                colour_print("red", "WARNING: The following result did not deregister the callback.")
        print(result)
        results.append(result)

    colour_print("green", "All tests have been re-evaluated.")
    if not y and not ask_yesno(
        info="Please inspect the evaluations carefully.",
        question="Do you wish to overwrite the result files?",
        default_yes=False,
    ):
        return
    for result in results:
        result.save(agent_name)
    colour_print("green", "Test results have been overwritten.")


if __name__ == "__main__":
    main()
