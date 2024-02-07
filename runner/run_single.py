import click
from dataset_interfaces.factory import DATASETS
from dataset_interfaces.interface import DatasetInterface
from runner.config import RunConfig
from runner.run_benchmark import get_chat_session, check_result_files
from runner.scheduler import TestRunner


@click.command('run-single')
@click.option('-a', '--agent', required=True, type=str)
@click.option('-d', '--datasets', required=True, type=str)
@click.option('-p', '--max-prompt-size', required=False, type=int, default=2000)
@click.option('-n', '--num-examples-per-dataset', required=False, type=int, default=1)
@click.option('-f', '--filler-tokens', required=False, type=int, default=100,
              help="Normal filler tokens")
@click.option('-qf', '--pre-question-filler-tokens', required=False, type=int,
              default=2000, help="Pre-question filler tokens")
def main(agent: str, datasets: str, max_prompt_size: int, num_examples_per_dataset: int,
         filler_tokens: int, pre_question_filler_tokens: int):
    print(f'Testing with a maximum prompt size of {max_prompt_size}, '
          f'{filler_tokens} normal filler tokens, and '
          f'{pre_question_filler_tokens} pre-question filler tokens.')
    dataset_list = datasets.split(",")
    dataset_list = [d.strip() for d in dataset_list]
    examples = []
    for ds_name in dataset_list:
        params = dict(
            filler_tokens_low=filler_tokens,
            filler_tokens_high=filler_tokens,
            pre_question_filler=pre_question_filler_tokens,
        )
        ds: DatasetInterface = DATASETS[ds_name](**params)
        examples.extend(
            ds.generate_examples(num_examples_per_dataset)
        )

    chat_session = get_chat_session(agent, max_prompt_size)

    conf = RunConfig(run_name="_RunSingle")
    check_result_files(conf.run_name, chat_session.name, force_removal=True)
    runner = TestRunner(
        config=conf,
        agent=chat_session,
        tests=examples,
    )
    runner.run()
    print("done")


if __name__ == '__main__':
    main()
