from pathlib import Path
from dataclasses import dataclass
from typing import List

from dataset_interfaces.gpt_generated import GPTGenerated
from dataset_interfaces.interface import TestExample
from utils.constants import DATA_DIR


@dataclass
class InstructionRecallDataset(GPTGenerated):
    name: str = "Instruction Recall"
    description: str = "Give the agent a list of instructions, then ask it multiple questions about these instructions"
    generation_file: Path = DATA_DIR.joinpath("gpt_generation_prompts/5-2_instruction_recall.json")
    reset_message: str = "Forget all of the instructions for operating the technology that I have given you up until this message."


if __name__ == "__main__":
    s = InstructionRecallDataset()
    examples = s.generate_examples(4)
    a = 1
