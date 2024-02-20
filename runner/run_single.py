import click
from dataset_interfaces.factory import DATASETS
from runner.config import RunConfig
from runner.run_benchmark import get_chat_session, check_result_files, generate_test_examples
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
@click.option("-y", required=False, is_flag=True, default=False, help="Automatically assent to questions")
def main(agent: str, datasets: str, max_prompt_size: int, num_examples_per_dataset: int,
         filler_tokens: int, pre_question_filler_tokens: int, y: bool):
    print(f'Testing with a maximum prompt size of {max_prompt_size}, '
          f'{filler_tokens} normal filler tokens, and '
          f'{pre_question_filler_tokens} pre-question filler tokens.')
    dataset_list = datasets.split(",")
    dataset_list = [d.strip() for d in dataset_list]
    chat_session = get_chat_session(agent, max_prompt_size)
    run_name = "_RunSingle"
    config_dict = dict(
        config=dict(
            run_name=run_name,
        ),
        datasets=dict(
            args=dict(
                filler_tokens_low=filler_tokens,
                filler_tokens_high=filler_tokens,
                pre_question_filler=pre_question_filler_tokens,
                dataset_examples=num_examples_per_dataset,
            ),
            datasets=[dict(name=d) for d in dataset_list],
        )
    )
    examples = generate_test_examples(config_dict,
                                      max_message_tokens=chat_session.max_message_size,
                                      pass_default=y,
                                      force_regenerate=True)
    check_result_files(run_name, chat_session.name, force_removal=True)
    yaml_config = config_dict["config"]
    config = {k: v for k, v in yaml_config.items() if k != "incompatibilities"}
    incompatibilities = []
    for inc_list in yaml_config.get("incompatibilities", []):
        incompatibilities.append([DATASETS[ds_name] for ds_name in inc_list])
    conf = RunConfig(incompatibilities=incompatibilities, **config)
    runner = TestRunner(
        config=conf,
        agent=chat_session,
        tests=examples,
    )
    runner.run()
    print("done")


if __name__ == '__main__':
    main()
