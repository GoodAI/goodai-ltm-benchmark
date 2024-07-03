import os

from dataset_interfaces.interface import TestExample, DatasetInterface
from local_datasets.conflicting_personal_information import ConflictingPersonalInformationDataset
from local_datasets.delayed_recall import DelayedRecallDataset
from local_datasets.how_to_think import HowToThinkDataset
from local_datasets.instruction_recall import InstructionRecallDataset
from local_datasets.prospective_memory import ProspectiveMemoryDataset
from local_datasets.colours import ColourDataset
from local_datasets.jokes import JokesDataset
from local_datasets.locations import LocationsDataset
from local_datasets.locations_directions import LocationsDirectionsDataset
from local_datasets.name import NamesDataset
from local_datasets.name_list import NameListDataset
from local_datasets.sally_ann import SallyAnneDataset
from local_datasets.shopping import ShoppingDataset
from local_datasets.spy_meeting import SpyMeetingDataset
from local_datasets.trigger_response import TriggerResponseDataset
from local_datasets.kv import KVPairsDataset
from local_datasets.chapterbreak import ChapterBreakDataset
from local_datasets.restaurant import RestaurantDataset
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