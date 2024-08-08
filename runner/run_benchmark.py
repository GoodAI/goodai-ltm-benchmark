import logging
import os.path
import os
import re
import shutil
import time
from typing import Optional

import click
import yaml

from pathlib import Path
from dataset_interfaces.factory import DatasetFactory, DATASETS
from dataset_interfaces.interface import TestExample
from model_interfaces.length_bias_agent import LengthBiasAgent
from model_interfaces.interface import ChatSession
from model_interfaces.llm_interface import LLMChatSession, TimestampLLMChatSession
from model_interfaces.ltm_agent_wrapper import LTMAgentWrapper
from model_interfaces.cost_estimation import CostEstimationChatSession
from model_interfaces.human import HumanChatSession
from model_interfaces.huggingface_interface import HFChatSession
from model_interfaces.gemini_interface import GeminiProInterface
from runner.config import RunConfig
from runner.scheduler import TestRunner
from utils.ui import ask_yesno, colour_print
from utils.files import gather_testdef_files, gather_result_files, make_run_path, make_config_path, \
    gather_persistence_files
from utils.llm import GPT_4_TURBO_BEST
from utils.constants import MAIN_DIR, TESTS_DIR


def get_chat_session(name: str, max_prompt_size: Optional[int], run_name: str, is_local=False) -> ChatSession:
    kwargs = {"max_prompt_size": max_prompt_size} if max_prompt_size is not None else {}
    kwargs["run_name"] = run_name
    kwargs["is_local"] = is_local

    if name == "gemini":
        return GeminiProInterface(run_name=run_name)

    if name.startswith("ltm_agent"):
        match = re.match(r"^ltm_agent\((?P<model>.+)\)$", name)
        if match is None:
            raise ValueError(f"Unrecognized LTM Agent {repr(name)}.")
        return LTMAgentWrapper(model=match.groupdict()["model"], **kwargs)
    if name == "length_bias":
        return LengthBiasAgent(model=GPT_4_TURBO_BEST, **kwargs)
    if name.startswith("cost("):
        in_cost, out_cost = [float(p.strip()) / 1_000 for p in name.removeprefix("cost(").removesuffix(")").split(",")]
        return CostEstimationChatSession(cost_in_token=in_cost, cost_out_token=out_cost, **kwargs)
    if name == "human":
        return HumanChatSession(**kwargs)
    if name.startswith("huggingface/"):
        kwargs.pop("is_local")
        return HFChatSession(model=name, **kwargs)

    try:
        if name.startswith("ts-"):
            cls = TimestampLLMChatSession
        else:
            cls = LLMChatSession

        return cls(model=name.removeprefix("ts-"), **kwargs)
    except ValueError:
        pass

    raise ValueError(f"Unrecognized agent: {name}")


def generate_test_examples(
    loaded_yaml, max_message_size: int, pass_default: bool = False, force_regenerate: bool = False
) -> list[TestExample]:
    run_name = loaded_yaml["config"]["run_name"]
    test_definitions = gather_testdef_files(run_name)

    if len(test_definitions) > 0:
        if not force_regenerate:
            if pass_default or ask_yesno(
                f"There are test definitions in disk for run name {run_name}",
                question="Do you want to reuse these test definitions?",
            ):
                return load_test_examples(loaded_yaml, test_definitions)
            if not ask_yesno(
                "WARNING: overwriting the test definitions will result in the loss of all "
                "results associated with them, including those from other agents.",
                default_yes=False,
            ):
                raise ValueError("Run aborted")
        shutil.rmtree(make_run_path(run_name))

    # Save original yaml configuration
    config_path = make_config_path(run_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as file:
        yaml.safe_dump(loaded_yaml, file)

    examples: list[TestExample] = []
    dataset_yaml = loaded_yaml["datasets"]
    for ds in dataset_yaml["datasets"]:
        examples.extend(DatasetFactory.create_examples(ds, dataset_yaml["args"], max_message_size))

    for example in examples:
        example.save(run_name)

    return examples


def load_test_examples(yaml_configuration, test_definition_paths: list[str]) -> list[TestExample]:

    examples = []
    for p in test_definition_paths:
        dataset = DatasetFactory.create_dataset_for_example(yaml_configuration, p)
        examples.append(TestExample.load(dataset, p))

    return examples


def check_result_files(run_name: str, agent_name: str, force_removal: bool = False, pass_default: bool = False):
    result_files = gather_result_files(run_name, agent_name)
    persistence_files = gather_persistence_files(run_name, agent_name)
    all_files = result_files + persistence_files
    resume = False
    if force_removal:
        for file in all_files:
            os.remove(file)
        all_files = []
    if len(all_files) > 0:
        if not pass_default:
            if not ask_yesno(
                f"There are {len(all_files)} existing file that have been found for run name '{run_name}' " f"and agent '{agent_name}'.",
                question="Do you want to resume the run?",
            ):
                if not ask_yesno(
                    "ALL RESULT FILES WILL BE LOST for the current run name and agent.",
                    default_yes=False,
                ):
                    colour_print("red", "Run aborted.")
                    exit()
                for file in all_files:
                    os.remove(file)
                resume = False
            else:
                resume = True
    return resume


@click.command("run-benchmark")
@click.option(
    "-c",
    "--configuration",
    required=False,
    type=str,
    default="./configurations/benchmark_1.yml",
)
@click.option("-a", "--agent-name", required=True, type=str)
@click.option("-m", "--max-prompt-size", required=False, type=int, default=None)
@click.option("-y", required=False, is_flag=True, default=False, help="Automatically assent to questions")
@click.option("-l", "--local", required=False, is_flag=True, default=False, help="Do not try to retrieve costs.")
@click.option("-i", "--isolated", required=False, is_flag=True, default=False, help=(
        "Run tests separately, without interleaving and clearing up the context between tests."
))
def main(
    configuration: str, agent_name: str, max_prompt_size: Optional[int], y: bool = False, local: bool = False,
    isolated: bool = False
):
    _main(configuration, agent_name, max_prompt_size, y, local, isolated)


def _main(
    configuration: str, agent_name: str, max_prompt_size: Optional[int], y: bool = False, is_local: bool = False,
    isolated: bool = False
):
    config_path = Path(configuration)
    if not config_path.is_absolute():
        config_path = MAIN_DIR.joinpath(configuration)
    with open(config_path, "rb") as file:
        loaded_yaml = yaml.safe_load(file)

    yaml_config = loaded_yaml["config"]
    config = {k: v for k, v in yaml_config.items() if k != "incompatibilities"}
    incompatibilities = []
    for inc_list in yaml_config.get("incompatibilities", []):
        incompatibilities.append({DATASETS[ds_name] for ds_name in inc_list})
    num_repetitions = loaded_yaml["datasets"]["args"]["dataset_examples"]
    conf = RunConfig(
        incompatibilities=incompatibilities, isolated=isolated, num_examples_per_dataset=num_repetitions, **config
    )
    if isolated:
        new_run_name = f"{conf.run_name} (isolated)"
        new_def_path = TESTS_DIR.joinpath(new_run_name, "definitions")
        if not new_def_path.exists():
            shutil.copytree(TESTS_DIR.joinpath(conf.run_name, "definitions"), new_def_path)
        conf.run_name = new_run_name
    if max_prompt_size is None:
        logging.warning("Running without a maximum prompt size.")
    else:
        print(f"Maximum prompt size: {max_prompt_size}")

    agent = get_chat_session(agent_name, max_prompt_size=max_prompt_size, run_name=config['run_name'], is_local=is_local)

    examples = generate_test_examples(loaded_yaml, agent.max_message_size, pass_default=y)
    resume = check_result_files(conf.run_name, agent.name, pass_default=y)
    if resume:
        agent.load()

    runner = TestRunner(config=conf, agent=agent, tests=examples, skip_evaluations=agent_name.startswith("cost("))
    time1 = time.time()
    runner.run()

    time2 = time.time()
    elapsed = (time2 - time1) / 60
    colour_print("green", f"Done in {elapsed:.2g} minutes.")


if __name__ == "__main__":
    main()
