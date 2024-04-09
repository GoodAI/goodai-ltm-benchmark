import click
from dataset_interfaces.factory import DATASETS
from reporting.results import TestResult
from utils.files import gather_result_files


@click.command("reevaluate")
@click.argument("dataset_key", type=str)
@click.option("-r", "--run-name", type=str, required=False, default="*")
@click.option("-a", "--agent-name", type=str, required=False, default="*")
def reevaluate(dataset_key: str, run_name: str = "*", agent_name: str = "*"):

    eval_stats = dict(cost=0)

    def cost_callback(cost_usd: float):
        eval_stats["cost"] += cost_usd

    ds = DATASETS[dataset_key](cost_callback=cost_callback)
    result_files = gather_result_files(run_name, agent_name, dataset_name=ds.name)
    for i, path in enumerate(result_files):
        percentage = (100 * (i + 1)) // len(result_files)
        print(f"\rRe-evaluating '{run_name}/results/{agent_name}': {percentage: 3d}%", end="")
        result = TestResult.from_file(path)
        evaluate = ds.evaluation_fn() 
        result.score, result.max_score, result.reasoning = evaluate(
            questions=[],
            responses=result.actual_responses,
            expected_answers=result.expected_responses,
        )
        result.save()

    print(f"\nReevaluation cost: {eval_stats['cost']}")


if __name__ == "__main__":
    reevaluate()
