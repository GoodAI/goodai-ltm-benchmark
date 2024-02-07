from dataclasses import dataclass
from random import choice
from typing import List, Tuple

from datasets.locations import (
    LocationsDataset,
    DISTANCES,
    DIRECTIONS,
)


@dataclass
class LocationsDirectionsDataset(LocationsDataset):
    name: str = "Locations Directions"
    question: str = "Given the points of interest that I have told you about, how would I travel from {{origin}} to {{place}} following those interesting points?"

    def generate_answer(self, location_info: List[Tuple[str, str, str, int]]) -> Tuple[Tuple[str, int], List[str]]:
        # Choose the direction, and distance - it doesn't really matter
        last_move = (choice(DIRECTIONS), choice(DISTANCES))

        # Now the answer should include all the directions and locations
        answer = ""
        for l in location_info[1:-1]:
            answer += f"From {l[1]}, go {l[3]}km {l[2]} to {l[0]}.\n"

        answer += f"From {location_info[-1][1]}, go {last_move[1]}km {last_move[0]} to {location_info[-1][0]}."

        return last_move, [answer]
