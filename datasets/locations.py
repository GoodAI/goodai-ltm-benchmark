from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

from dataset_interfaces.interface import DatasetInterface, TestExample


LOCATIONS = [
    "Town Hall",
    "Statue of Founder",
    "High Street",
    "Retail Area",
    "Playground",
    "Park",
    "Hospital",
    "Library",
    "Museum",
]
DIRECTIONS = [
    "North",
    "South",
    "East",
    "West",
]

DISTANCES = [1, 2, 3, 4]

STATEMENTS = [
    "{place} is a location in my home town, it is {distance} KM, {direction} from {other_place}.",
    "I visited {place}, which is {distance} KM, {direction} from {other_place}.",
    "About {distance} km {direction} of the {other_place} there is a {place}",
]


@dataclass
class LocationsDataset(DatasetInterface):
    name: str = "Locations"
    description: str = (
        "Tell the agent about locations in our hometown, with each position being described relative to the previous "
        "one. Finally the agent is asked the distance and direction from the first mentioned place to the last."
    )
    question: str = (
        "Please calculate the direction and distance starting from {place} going directly to {origin}. Solve it "
        "step by step."
    )
    known_locations: int = 5
    reset_message: str = (
        "Forget, or otherwise disregard, all of the points of interest from my home town that I have told you about "
        "before this message."
    )

    def generate_examples(self, num_examples):
        examples = []
        for _ in range(num_examples):
            script = []
            is_question = []
            known_locations = []
            location_information = []
            locations = deepcopy(LOCATIONS)

            for change in range(self.known_locations - 1):
                place = self.random.choice(locations)
                locations.remove(place)

                if len(known_locations) == 0:
                    statement = "There is a {place} in the center of my hometown."
                    other_place = ""
                    direction = ""
                    distance = 0
                else:
                    statement = self.random.choice(STATEMENTS)
                    other_place = known_locations[-1]
                    direction = self.random.choice(DIRECTIONS)
                    distance = self.random.choice(DISTANCES)

                known_locations.append(place)
                location_information.append((place, other_place, direction, distance))

                script.append(
                    statement.format(
                        place=place,
                        other_place=other_place,
                        direction=direction,
                        distance=distance,
                    )
                )
                is_question.append(False)

            # The current point is some distance and direction away from the origin.
            # Distance is fine, but we should align the direction correctly to NSEW
            place = self.random.choice(locations)
            other_place = known_locations[-1]
            location_information.append((place, other_place, "", 0))
            last_move, answer_list = self.generate_answer(location_information)
            script.append(
                f"And finally there is {place} which is {last_move[0]} from {other_place} and {last_move[1]} KM from it.",
            )
            is_question.append(False)

            origin = known_locations[0]
            question = self.question.format(place=place, origin=origin)

            script.append(question)
            is_question.append(True)

            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=answer_list,
                is_question=is_question,
            )

            examples.append(example)

        return examples

    def evaluate_correct(
        self,
        questions: List[str],
        responses: List[str],
        expected_answers: List[str],
    ) -> Tuple[int, int, List[str]]:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)

    def apply_move(self, current_position: list[int], direction: str, distance: int) -> list[int]:
        x = current_position[0]
        y = current_position[1]

        if direction == "North":
            y += distance
        elif direction == "South":
            y -= distance
        elif direction == "East":
            x += distance
        elif direction == "West":
            x -= distance

        return [x, y]

    def generate_answer(
        self, location_info: List[Tuple[str, str, str, int]]
    ) -> Tuple[Tuple[str, int], List[str]]:
        # Follow the location
        # Position the current location on a 2d plane centered around the origin.
        # Generate an answer so that the final location will be (x, 0) or (0, y)
        current_pos = [0, 0]

        for l in location_info[1:-1]:
            current_pos = self.apply_move(current_pos, l[2], l[3])

        # If we are not currently aligned along an axis
        if current_pos[0] != 0 and current_pos[1] != 0:
            # Pick an axis and align it
            idx = self.random.randint(0, 1)
            if idx == 0:
                # East/West - this is the direction that we want to go
                direction = "East" if current_pos[idx] < 0 else "West"
            elif idx == 1:
                # North/South - this is the direction that we want to go
                direction = "North" if current_pos[idx] < 0 else "South"

            distance = abs(current_pos[idx])

        elif current_pos[1] == 0:
            # We are currently aligned along East/West
            east_west_pos = current_pos[0]
            distance = (
                self.random.randint(-4, abs(east_west_pos) - 1)
                if east_west_pos < 0
                else self.random.randint(-(east_west_pos - 1), 4)
            )
            direction = "East" if distance > 0 else "West"
            distance = abs(distance)

        elif current_pos[0] == 0:
            # We are currently aligned along North/South
            north_south_pos = current_pos[1]
            distance = (
                self.random.randint(-4, abs(north_south_pos - 1))
                if north_south_pos < 0
                else self.random.randint(-(north_south_pos - 1), 4)
            )
            direction = "North" if distance > 0 else "South"
            distance = abs(distance)

        last_move = (direction, distance)
        current_pos = self.apply_move(current_pos, direction, distance)
        answer = self.to_origin(current_pos)

        return last_move, answer

    def to_origin(self, current_pos):
        if current_pos[0] == 0 and current_pos[1] > 0:
            direction = "South"
            distance = abs(current_pos[1])
        elif current_pos[0] == 0 and current_pos[1] < 0:
            direction = "North"
            distance = abs(current_pos[1])
        elif current_pos[1] == 0 and current_pos[0] > 0:
            direction = "West"
            distance = abs(current_pos[0])
        elif current_pos[1] == 0 and current_pos[0] < 0:
            direction = "East"
            distance = abs(current_pos[0])
        else:  # Coordinates are (0,0)
            return "They are in the same location."

        return [f"{distance} km {direction}"]

