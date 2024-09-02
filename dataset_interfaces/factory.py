import os

from dataset_interfaces.interface import TestExample, DatasetInterface
from datasets.code_defs import CodeDefinitionsDataset
from datasets.conficting_personal_information import ConflictingPersonalInformationDataset
from datasets.delayed_recall import DelayedRecallDataset
from datasets.how_to_think import HowToThinkDataset
from datasets.instruction_recall import InstructionRecallDataset
from datasets.lb_wiki import LongBenchWikiQADataset
from datasets.prospective_memory import ProspectiveMemoryDataset
from datasets.colours import ColourDataset
from datasets.jokes import JokesDataset
from datasets.locations import LocationsDataset
from datasets.locations_directions import LocationsDirectionsDataset
from datasets.name import NamesDataset
from datasets.name_list import NameListDataset
from datasets.sally_ann import SallyAnneDataset
from datasets.shopping import ShoppingDataset
from datasets.spy_meeting import SpyMeetingDataset
from datasets.trigger_response import TriggerResponseDataset
from datasets.kv import KVPairsDataset
from datasets.chapterbreak import ChapterBreakDataset
from datasets.restaurant import RestaurantDataset
from copy import deepcopy

from utils.files import parse_definition_path

DATASETS = {
    "names": NamesDataset,
    "colours": ColourDataset,
    "shopping": ShoppingDataset,
    "locations": LocationsDataset,
    "locations_directions": LocationsDirectionsDataset,
    "sallyanne": SallyAnneDataset,
    "name_list": NameListDataset,
    "jokes": JokesDataset,
    "kv": KVPairsDataset,
    "chapterbreak": ChapterBreakDataset,
    "delayed_recall": DelayedRecallDataset,
    "instruction_recall": InstructionRecallDataset,
    "how_to_think": HowToThinkDataset,
    "prospective_memory": ProspectiveMemoryDataset,
    "conflicting_personal_info": ConflictingPersonalInformationDataset,
    "trigger_response": TriggerResponseDataset,
    "restaurant": RestaurantDataset,
    "spy_meeting": SpyMeetingDataset,
    "wiki_qa": LongBenchWikiQADataset,
    "code_defs": CodeDefinitionsDataset
}
DATASETS_BY_NAME = {ds.name: ds for ds in DATASETS.values()}


class DatasetFactory:
    @staticmethod
    def create_examples(dataset_config: dict, universal_args: dict, max_message_size: int) -> list[TestExample]:
        name = dataset_config["name"]
        args = deepcopy(universal_args)
        ds_args = dataset_config.get("args", {})
        args.update(ds_args)
        num_examples = args["dataset_examples"]
        del args["dataset_examples"]
        ds_factory = DATASETS.get(name)
        if not ds_factory:
            raise ValueError(f"No such dataset: {name}")
        ds = ds_factory(max_message_size=max_message_size, **args)
        examples = ds.generate_examples(num_examples)
        for i, example in enumerate(examples):
            if example.example_id == "":
                example.example_id = str(i)
        return examples

    @staticmethod
    def create_dataset_for_example(run_configuration: dict,  test_example_path: str) -> DatasetInterface:
        args = deepcopy(run_configuration["datasets"]["args"])

        # Get the name of the dataset
        path_dataset_name = parse_definition_path(test_example_path)["dataset_name"]
        dataset = DATASETS_BY_NAME.get(path_dataset_name, None)

        if dataset is None:
            raise ValueError(f"No dataset could be resolved from TestExample path: {test_example_path}. Tried {path_dataset_name}")

        # Use the class to get the 'config name'
        config_name = ""
        for name, ds in DATASETS.items():
            if ds == dataset:
                config_name = name
                break

        # Use config name to get any extra config from the run configuration
        extra_args = {}
        for dataset_config in run_configuration["datasets"]["datasets"]:
            if dataset_config["name"] == config_name:
                extra_args = dataset_config.get("args", {})
                break

        args.update(extra_args)
        del args["dataset_examples"]
        return dataset(**args)