import re
from json import JSONDecodeError
from dataclasses import dataclass
from typing import List, Tuple
from utils.openai import make_user_message
from utils.json_utils import LLMJSONError
from goodai.helpers.json_helper import sanitize_and_parse_json
from datasets.locations import (
    LocationsDataset,
    LOCATIONS,
    DISTANCES,
    DIRECTIONS,
)

DirectionDict = dict[str, str | int]


@dataclass
class LocationsDirectionsDataset(LocationsDataset):
    name: str = "Locations Directions"
    question: str = (
        "Given the points of interest that I have told you about, how would I travel from {{origin}} to {{place}} "
        "following those interesting points?"
    )

    def generate_answer(self, location_info: List[Tuple[str, str, str, int]]) -> Tuple[Tuple[str, int], List[str]]:
        # Choose the direction, and distance - it doesn't really matter
        last_move = (self.random.choice(DIRECTIONS), self.random.choice(DISTANCES))

        # Now the answer should include all the directions and locations
        directions = ["(Just an example, if exact instructions are followed)"]
        for destination, origin, direction, distance in location_info[1:-1]:
            directions.append(f"From {origin}, go {distance}km {direction} to {destination}.")

        directions.append(f"From {location_info[-1][1]}, go {last_move[1]}km {last_move[0]} to {location_info[-1][0]}.")
        answer = "\n".join(directions)

        return last_move, [answer]

    def parse_directions(self, expected_answer: str) -> list[DirectionDict]:
        pattern = r"^From (?P<origin>\w+(?: \w+){0,2}), go (?P<distance>\d)km (?P<direction>\w+) to (?P<destination>\w+(?: \w+){0,2}).$"
        # Ignore first line and parse the rest
        directions = [re.match(pattern, line).groupdict() for line in expected_answer.splitlines()[1:]]
        for d in directions:
            d["distance"] = int(d["distance"])
        return directions

    def structure_directions(self, agent_response: str) -> list[DirectionDict]:
        allowed_locations = [loc.lower() for loc in LOCATIONS]
        allowed_directions = [d.lower() for d in DIRECTIONS]
        context = [make_user_message(structured_directions_prompt.format(
            directions=agent_response, places="\n".join(f"- {loc}" for loc in LOCATIONS),
        ))]
        response = self.ask_llm(context)
        try:
            directions = sanitize_and_parse_json(response)
            assert isinstance(directions, list)
            for d in directions:
                for k in ["origin", "destination"]:
                    assert d[k].lower() in allowed_locations, f"Location {repr(d[k])} is unknown."
                    d[k] = LOCATIONS[allowed_locations.index(d[k].lower())]
                assert d["direction"].lower() in allowed_directions, f"Direction {repr(d[k])} is unknown."
                d["direction"] = DIRECTIONS[allowed_directions.index(d["direction"].lower())]
                assert isinstance(d["kilometers"], int) and d["kilometers"] > 0, f"{d['kilometers']} is not a positive int."
                d["distance"] = d.pop("kilometers")
        except (JSONDecodeError, ValueError, KeyError, AssertionError) as exc:
            raise LLMJSONError(
                f"Couldn't make sense of the agent's directions ({repr(exc)}).\n"
                f"Original response:\n{agent_response}\n\nStructured version:\n{response}"
            )
        return directions

    def follow_directions(self, directions: list[DirectionDict], x0: int = 0, y0: int = 0) -> list[int]:
        pos = [x0, y0]
        for d in directions:
            axis = 0 if d["direction"].lower() in ["west", "east"] else 1
            sign = 1 if d["direction"].lower() in ["north", "east"] else -1
            pos[axis] += sign * d["distance"]
        return pos

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str],
    ) -> Tuple[int, int, List[str]]:
        score = 0
        max_score = 1
        exact_directions = self.parse_directions(expected_answers[0])
        expected_pos = self.follow_directions(exact_directions)
        try:
            agent_directions = self.structure_directions(responses[0])
            final_pos = self.follow_directions(agent_directions)
            score = int(final_pos == expected_pos)
            not_str = "" if score else "do not "
            reasoning = f"The agent's directions {not_str}lead to the expected destination."
        except LLMJSONError as exc:
            reasoning = str(exc)
        return score, max_score, [reasoning]


structured_directions_prompt = """
Take a look at this text:
```text
{directions}
```

Now convert it to a sequence of directions in a well-structured JSON, like this:
[
  {{"origin": "some place", "kilometers": 2, "direction": "West", "destination": "other place"}},
  ...
]

Also, if any place matches a place from this list, you must use the name from the list instead of what's in the text:
{places}
""".strip()
