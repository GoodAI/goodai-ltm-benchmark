from datasets.trigger_response import TriggerResponseDataset
from reporting.results import TestResult
from utils.files import gather_result_files


def reevaluate(run_name: str = "*", agent_name: str = "*"):

    eval_stats = dict(cost=0)

    def cost_callback(cost_usd: float):
        eval_stats["cost"] += cost_usd

    ds = TriggerResponseDataset(cost_callback=cost_callback)
    result_files = gather_result_files(run_name, agent_name, dataset_name=ds.name)
    for i, path in enumerate(result_files):
        percentage = (100 * (i + 1)) // len(result_files)
        print(f"\rRe-evaluating '{run_name}/{agent_name}': {percentage: 3d}%", end="")
        result = TestResult.from_file(path)
        result.score, result.max_score, result.reasoning = ds.evaluate_correct(
            questions=[],
            responses=result.actual_responses,
            expected_answers=result.expected_responses
        )
        result.save()

    print(f"\nReevaluation cost: {eval_stats['cost']}")


if __name__ == "__main__":
    reevaluate("Benchmark 2 - 1k Filler")
    reevaluate("Benchmark 2 - 10k Filler")
