from glob import glob
from pathlib import Path

from utils.constants import TESTS_DIR


def make_run_path(run_name: str, agent_name: str | None = None) -> Path:
    if agent_name is None:
        return TESTS_DIR.joinpath(f"{run_name}")
    return TESTS_DIR.joinpath(f"{run_name}/results/{agent_name}")


def make_config_path(run_name: str) -> Path:
    return make_run_path(run_name).joinpath("definitions/config.yml")


def make_runstats_path(run_name: str, agent_name: str) -> Path:
    return make_run_path(run_name, agent_name).joinpath("runstats.json")


def make_master_log_path(run_name: str, agent_name: str) -> Path:
    return make_run_path(run_name, agent_name).joinpath("master_log.jsonl")


def make_testdef_path(run_name: str, dataset_name: str, example_id: str) -> Path:
    return TESTS_DIR.joinpath(f"{run_name}/definitions/{dataset_name}/{example_id}.def.json")


def make_result_path(run_name: str, agent_name: str, dataset_name: str, example_id: str, repetition: int | str) -> Path:
    return make_run_path(run_name, agent_name).joinpath(f"{dataset_name}/{example_id}_{repetition}.json")


def gather_testdef_files(run_name: str = "*", dataset_name: str = "*", example_id: str = "*") -> list[str]:
    return glob(str(make_testdef_path(run_name, dataset_name, example_id)))


def gather_result_files(run_name: str = "*", agent_name: str = "*", dataset_name: str = "*") -> list[str]:
    return glob(str(make_result_path(run_name, agent_name, dataset_name, "*", "*")))


def gather_persistence_files(run_name: str = "*", agent_name: str = "*") -> list[str]:
    return glob(str(make_runstats_path(run_name, agent_name))) + glob(str(make_master_log_path(run_name, agent_name)))


def gather_runstats_files(run_name: str = "*", agent_name: str = "*") -> list[str]:
    return glob(str(make_runstats_path(run_name, agent_name)))


def get_run_names() -> list[str]:
    return [Path(path).name for path in glob(str(TESTS_DIR.joinpath("*"))) if Path(path).is_dir()]


def parse_result_path(path: Path | str) -> dict[str, str]:
    run_name, _, agent_name, dataset_name, result_name = Path(path).as_posix().split("/")[-5:]
    name_parts = result_name.removesuffix(".json").split("_")
    return dict(
        run_name=run_name,
        agent_name=agent_name,
        dataset_name=dataset_name,
        example_id="_".join(name_parts[:-1]),
        repetition=int(name_parts[-1])
    )
