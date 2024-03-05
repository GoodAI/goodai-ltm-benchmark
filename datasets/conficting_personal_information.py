from dataclasses import dataclass
from pathlib import Path
from dataset_interfaces.gpt_generated import GPTGenerated
from utils.constants import DATA_DIR


@dataclass
class ConflictingPersonalInformationDataset(GPTGenerated):
    name: str = "Conflicting Personal Information"
    description: str = "Assess how the LLM reconciles and responds to conflicting information."
    generation_file: Path = DATA_DIR.joinpath("gpt_generation_prompts/4-2_conflicting_personal_information_test.json")


if __name__ == "__main__":
    s = ConflictingPersonalInformationDataset()
    examples = s.generate_examples(4)
    a = 1
