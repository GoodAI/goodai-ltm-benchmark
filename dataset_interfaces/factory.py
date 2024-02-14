from typing import Optional
from datasets.conficting_personal_information import ConflictingPersonalInformationDataset
from datasets.delayed_recall import DelayedRecallDataset
from datasets.how_to_think import HowToThinkDataset
from datasets.instruction_recall import InstructionRecallDataset
from datasets.prospective_memory import ProspectiveMemoryDataset
from datasets.colours import ColourDataset
from datasets.jokes import JokesDataset
from datasets.locations import LocationsDataset
from datasets.locations_directions import LocationsDirectionsDataset
from datasets.name import NamesDataset
from datasets.name_list import NameListDataset
from datasets.sally_ann import SallyAnneDataset
from datasets.shopping import ShoppingDataset
from datasets.trigger_response import TriggerResponseDataset
from datasets.kv import KVPairsDataset
from datasets.chapterbreak import ChapterBreakDataset
from copy import deepcopy

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
}
DATASETS_BY_NAME = {ds.name: ds for ds in DATASETS.values()}


class DatasetFactory:
    @staticmethod
    def create_examples(dataset_config, universal_args):
        name = dataset_config["name"]
        args = deepcopy(universal_args)
        ds_args = dataset_config.get("args", {})
        args.update(ds_args)
        num_examples = args["dataset_examples"]
        del args["dataset_examples"]
        ds = DATASETS[name](**args)
        examples = ds.generate_examples(num_examples)
        for i, example in enumerate(examples):
            if example.example_id == "":
                example.example_id = str(i)
        return examples
