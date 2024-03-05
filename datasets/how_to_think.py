from pathlib import Path
from dataclasses import dataclass
from dataset_interfaces.gpt_generated import GPTGenerated
from utils.constants import DATA_DIR


@dataclass
class HowToThinkDataset(GPTGenerated):
    name: str = "How to Think"
    description: str = (
        "Evaluate the LLM's ability to apply a given reasoning or thought process to a particular situation or problem."
    )
    generation_file: Path = DATA_DIR.joinpath("gpt_generation_prompts/7-1_teaching_how_to_think.json")


if __name__ == "__main__":
    s = HowToThinkDataset()
    examples = s.generate_examples(4)
    a = 1
