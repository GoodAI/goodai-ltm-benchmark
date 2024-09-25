import json
import os
import random
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Any, Tuple

from openai import OpenAI

from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.constants import DATA_DIR

# Direct Questions:
## Who did <actor> play in <film>?  <character>
## What was the director of <film>? <director>
## In what year was <film> released? <year>
## Who was the actor that played <character>? <actor>
## Who was the director of the film in which <actor> played <character>? <director>
## <actor> played <character> in which film? <film>


# Whole Dataset Questions:
## How many films were released in <year>?
## Which director does <actor> work with the most


# Indirect Questions:
## Which film by <director> did the actor who has played <character> also star in?  (2 retrievals)
## What is the genre of the film was that was released first? The one by director <director> in which <actor> plays <character>, or the one with <shortened synopsis>?
## What <year> film did the

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class QuestionStructure(Enum):
    DIRECTOR = 0
    FILM = 1
    ACTOR = 2
    CHARACTER = 3
    COUNT = 4
    YEAR = 5
    TOTAL = 6



@dataclass
class MoviesDataset(DatasetInterface):
    name: str = "Movies"
    description: str = "Give the agent some data about fake movies. Ask questions that are direct, indirect, or about the whole dataset."
    type: str = "ALL" #, "WHOLE", "DIRECT", "INDIRECT"
    questions_per_example: int = 3
    reset_message = "Forget or otherwise disregard all of the movies that I have talked about before this point in the conversation."

    def generate_examples(self, num_examples: int) -> List[TestExample]:

        examples = []
        for _ in range(num_examples):

            # Load a sample of the data in proportion to the number of questions we will ask
            dataset = self.load_movies(self.questions_per_example * 7)
            statistics = self.gather_stats(dataset)
            questions = []

            for x in range(self.questions_per_example):
            # Generate questions first
                q_type = self.get_type()
                if q_type == "WHOLE":
                    q = self.generate_whole(statistics)
                    self.whole_natural_language(q)
                elif q_type == "DIRECT":
                    q = self.generate_direct(dataset)
                    self.direct_questions_natural_language(q)
                elif q_type == "INDIRECT":
                    q = self.generate_indirect(dataset)
                    self.indirect_questions_natural_language(q)
                else:
                    raise NotImplemented("Bad question type")

                questions.append(q)

            script = []
            is_question = []
            expected_responses = []

            movies_per_message = 3
            idx1 = 0
            idx2 = 0

            while idx2 < len(dataset) -1:
                idx2 = min(idx1 + movies_per_message, len(dataset) - 1)
                script_message = f"Here is some movie information that you might be interested in:\n{json.dumps(dataset[idx1:idx2], ensure_ascii=False, indent=2)}"
                idx1 = idx2

                script.append(script_message)
                is_question.append(False)

            for q in questions:
                script.append(q["question"])
                is_question.append(True)
                expected_responses.append(str(q['target_value']))

            examples.append(TestExample(self, script, expected_responses, is_question=is_question, script_is_filler=True))

        return examples

    def evaluate_correct(self, questions: List[str], responses: List[str], expected_answers: List[Any]) -> Tuple[int, int, List[str]]:
        score = 0
        reasoning = []

        for response, expected in zip(responses, expected_answers):
            if expected in response:
                score += 1
                reasoning.append("Answer is correct")
            else:
                reasoning.append("Answer is incorrect")

        return score / len(expected_answers), 1, reasoning


    def get_type(self):
        if self.type != "ALL":
            return self.type

        return self.random.choice(["WHOLE", "DIRECT", "INDIRECT"])

    def load_movies(self, num_movies):

        with open(DATA_DIR.joinpath("movies", "movies_list.json"), "r") as fp:
            all_data =  json.load(fp)

        return self.random.sample(all_data, k=num_movies)

    def gather_stats(self, dataset):
        length = len(dataset)
        movies_by_director = {}
        movies_by_actor = {}
        characters_by_actor = {}
        movies_by_year = {}
        movies_by_genre = {}

        stats = {"length": length}

        actors = set()
        characters = set()
        directors = set()
        genres = set()
        years = set()

        for film in dataset:
            # Easy stats
            actors.update(film["cast"])
            characters.update(film["characters"])
            directors.add(film["director"])
            genres.add(film["genre"])
            years.add(film["year"])

            if film["director"] not in movies_by_director.keys():
                movies_by_director[film["director"]] = []
            movies_by_director[film["director"]].append(film["title"])

            if film["year"] not in movies_by_year.keys():
                movies_by_year[film["year"]] = []
            movies_by_year[film["year"]].append(film["title"])

            if film["genre"] not in movies_by_genre.keys():
                movies_by_genre[film["genre"]] = []
            movies_by_genre[film["genre"]].append(film["title"])

            for idx, actor in enumerate(film["cast"]):
                if actor not in movies_by_actor.keys():
                    movies_by_actor[actor] = []
                movies_by_actor[actor].append(film["title"])

                if actor not in characters_by_actor.keys():
                    characters_by_actor[actor] = set()
                characters_by_actor[actor].add(film["characters"][idx])

        stats["actors"] = list(actors)
        stats["characters"] = list(characters)
        stats["directors"] = list(directors)
        stats["movies_by_director"] = movies_by_director
        stats["movies_by_actor"] = movies_by_actor
        stats["characters_by_actor"] = characters_by_actor
        stats["years"] = list(years)
        stats["genres"] = list(genres)
        stats["movies_by_year"] = movies_by_year
        stats["movies_by_genre"] = movies_by_genre
        return stats

    def generate_direct(self,
        dataset, target_attribute: Optional[str] = None, answer: Optional[str] = None
    ):
        print("Generating direct question")

        # Pathway 1: The target is not specified, so we randomly pick one and use that
        if not target_attribute:
            # Pick a random film that will contain our answer.
            film = self.random.choice(dataset)
            # Pick a target attribute of the film
            allowed_target_attributes = list(film.keys())
            # Synopsis cannot be a target attribute
            allowed_target_attributes.remove("synopsis")

            target_attribute = self.random.choice(allowed_target_attributes)

            if target_attribute not in ["characters", "cast"]:
                answer = film[target_attribute]
            else:
                answer = self.random.choice(film[target_attribute])
        # Pathway 2: We have a target attribute and value in mind, so find a film that satisfies those criteria
        else:
            attr_search = {target_attribute: answer}
            search_results = self.search_dataset(dataset, attr_search)
            film = self.random.choice(search_results)

        search_attributes = list(film.keys())
        # We are not going to give the answer to the task in the question
        search_attributes.remove(target_attribute)

        attributes_selection = {}

        # Character and cast attributes are linked, so if one of them is the target, the other has to be a search_attribute
        if target_attribute == "characters":
            target_attribute = "character"
            search_attributes.remove("cast")
            attributes_selection["cast"] = film["cast"][film["characters"].index(answer)]

        if target_attribute == "cast":
            search_attributes.remove("characters")
            attributes_selection["characters"] = film["characters"][
                film["cast"].index(answer)
            ]

        search_results = self.search_dataset(dataset, attributes_selection)
        if len(search_results) < 1:
            raise ValueError(
                f"Something broke generation and search using attributes {attributes_selection} to obtain {film}"
            )

        unique = len(search_results) == 1
        while not unique:
            attribute, value = self.get_search_attribute(film, search_attributes)
            search_attributes.remove(attribute)
            attributes_selection[attribute] = value

            search_results = self.search_dataset(dataset, attributes_selection)
            if len(search_results) < 1:
                raise ValueError(
                    f"Something broke generation and search using attributes {attributes_selection} to obtain {film}"
                )

            unique = len(search_results) == 1

        question_structure = {
            "title": film["title"],
            "target_attribute": target_attribute,
            "target_value": answer,
            "clues": attributes_selection,
        }

        return question_structure

    def search_dataset(self, dataset, search_attributes):
        results = []
        for film in dataset:
            include = True
            for attribute, value in search_attributes.items():
                if isinstance(film[attribute], list):
                    if value not in film[attribute]:
                        include = False
                        break

                elif film[attribute] != value:
                    include = False
                    break

            if include:
                results.append(film)

        return results

    def get_search_attribute(self, film, allowed_attributes):
        selection_attribute = self.random.choice(allowed_attributes)
        selection_value = film[selection_attribute]

        if isinstance(selection_value, list):
            selection_value = self.random.choice(selection_value)

        return selection_attribute, selection_value

    def direct_questions_natural_language(self, question):
        q_dict = {
            "target_attribute": question["target_attribute"],
            "clues": question["clues"],
        }

        model = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""You are to generate a quiz question about about a film based on the above supplied attribute. Phrase it like a question and follow these rules:   
For "cast" clues, mention that the film stars this person.
For "character" clues mention that the film features a character with that name.
For synopsis, shorten the plot to at most seven words. Do not mention the title, cast, or characters.

The "target_attribute" is what the question should ask.
{json.dumps(q_dict, indent=4)}""",
                },
            ],
        )

        question["question"] = response.choices[0].message.content

    def generate_whole(self, stats):
        # For the whole dataset, we generate comparisons or counts
        metric = ["compare", "count"]
        # Lets generate a question
        question_struct = {}
        sel_metric = self.random.choice(metric)
        if sel_metric == "compare":
            sel_attribute = self.random.choice(["directors", "actors", "years", "genres"])
            attrs_to_pick_from = deepcopy(list(stats[sel_attribute]))

            # We have what we are comparing, the main thing we are comparing is "films" and criteria is who/what had more?
            compared_value1 = self.random.choice(attrs_to_pick_from)
            attrs_to_pick_from.remove(compared_value1)
            compared_value2 = self.random.choice(attrs_to_pick_from)
            attrs_to_pick_from.remove(compared_value2)

            # Get the answer
            if sel_attribute == "directors":
                if len(stats["movies_by_director"][compared_value1]) > len(
                    stats["movies_by_director"][compared_value2]
                ):
                    answer = compared_value1
                elif len(stats["movies_by_director"][compared_value2]) > len(
                    stats["movies_by_director"][compared_value1]
                ):
                    answer = compared_value2
                else:
                    answer = "same"

            elif sel_attribute == "actors":
                if len(stats["movies_by_actor"][compared_value1]) > len(
                    stats["movies_by_actor"][compared_value2]
                ):
                    answer = compared_value1
                elif len(stats["movies_by_actor"][compared_value2]) > len(
                    stats["movies_by_actor"][compared_value1]
                ):
                    answer = compared_value2
                else:
                    answer = "same"

            elif sel_attribute == "years":
                if len(stats["movies_by_year"][compared_value1]) > len(
                    stats["movies_by_year"][compared_value2]
                ):
                    answer = compared_value1
                elif len(stats["movies_by_year"][compared_value2]) > len(
                    stats["movies_by_year"][compared_value1]
                ):
                    answer = compared_value2
                else:
                    answer = "same"

            else:  # sel_attribute == "genres":
                if len(stats["movies_by_genre"][compared_value1]) > len(
                    stats["movies_by_genre"][compared_value2]
                ):
                    answer = compared_value1
                elif len(stats["movies_by_genre"][compared_value2]) > len(
                    stats["movies_by_genre"][compared_value1]
                ):
                    answer = compared_value2
                else:
                    answer = "same"

            question_struct["type"] = "compare"
            question_struct["target_value"] = answer
            question_struct["attribute"] = sel_attribute
            question_struct["compared1"] = compared_value1
            question_struct["compared2"] = compared_value2

        else:  # sel_metric == "count"
            sel_attribute = self.random.choice(["directors", "actors", "years", "genres"])
            if self.random.random() < 0.5:
                instance = "all"
                answer = len(stats[sel_attribute])
            else:
                instance = self.random.choice(stats[sel_attribute])
                if sel_attribute == "directors":
                    answer = len(stats["movies_by_director"][instance])

                elif sel_attribute == "actors":
                    answer = len(stats["movies_by_actor"][instance])

                elif sel_attribute == "years":
                    answer = len(stats["movies_by_year"][instance])

                else:
                    answer = len(stats["movies_by_genre"][instance])

            question_struct["type"] = "count"
            question_struct["attribute"] = sel_attribute
            question_struct["instance"] = instance
            question_struct["target_value"] = answer

        return question_struct

    def whole_natural_language(self, q_dict):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""Here is a question about a dataset of films as structured data. Phrase the question in natural language.
A "count" question will count the number of films as according to the selected attribute and specific instance.

If the instance is "all", then the question should be about how many of that attribute (e.g directors, actors, years) are in the dataset.

A compare question is to return which instance of the attribute has more films. But phrase it more naturally according to the type of relationship that attribute would have with a film. Make sure you mention each of the clues.

        {json.dumps(q_dict, indent=4)}""",
                },
            ],
        )

        q_dict["question"] = response.choices[0].message.content

    def generate_indirect(self, dataset):
        invalid = True

        while invalid:
            initial_question = self.generate_direct(dataset)

            # Some clues will be unique to the film and are unsuitable for indirect questions
            valid_clues = deepcopy(initial_question["clues"])
            for key in list(valid_clues.keys()):
                if key in ["title", "synopsis", "characters"]:
                    del valid_clues[key]

            if len(valid_clues) == 0:
                continue

            # Lets do a single clue
            indirect_target_attribute = self.random.choice(list(valid_clues.keys()))

            second_question = self.generate_direct(
                dataset,
                indirect_target_attribute,
                initial_question["clues"][indirect_target_attribute],
            )

            # A film cannot be used to reference itself
            if initial_question["title"] == second_question["title"]:
                continue
            else:
                invalid = False

        initial_question["clues"][indirect_target_attribute] = second_question
        return initial_question

    def indirect_questions_natural_language(self, question):
        model = "gpt-4o-mini"

        indirect_clue_attribute = [
            c for c in question["clues"].keys() if type(question["clues"][c]) is dict
        ][0]

        # Split out the inner question (Q2)
        q_2_dict = {
            "target_attribute": question["clues"][indirect_clue_attribute][
                "target_attribute"
            ],
            "clues": question["clues"][indirect_clue_attribute]["clues"],
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""You are to generate a quiz question to obtain information on the supplied attribute of a film. Phrase it like a question and follow these rules:   
                - Use all the clues
                - For "cast" clues, mention that this person is an actor.
                - For "character" clues mention that there is a character with that name.
                - For "synopsis" clues, shorten the plot to at most seven words. Do not mention the title, cast, or characters.
                - For "title" clues, state the title, and say that it is the title.

                Use phrasing like:
                - "In which year was ..."
                - "Which actor portrays..."
                - "Who directed ..."
                - "What is the title of..."


                The "target_attribute" is what the question should ask.
        {json.dumps(q_2_dict, indent=4)}""",
                },
            ],
        )

        question2 = response.choices[0].message.content

        # Now create a question for the main part, remove
        other_clues = deepcopy(question["clues"])
        del other_clues[indirect_clue_attribute]
        q_dict = {
            "target_attribute": question["target_attribute"],
            "clues": other_clues,
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""You are to generate a quiz question to obtain information on the supplied attribute of a film. Phrase it like a question and follow these rules:   
                - Use all the clues
                - For "cast" clues, mention that this person is an actor.
                - For "character" clues mention that there is a character with that name.
                - For "synopsis" clues, shorten the plot to at most seven words. Do not mention the title, cast, or characters.
                - For "title" clues, state the title, and say that it is the title.

                Use phrasing like:
                - "In which year was ..."
                - "Which actor portrays..."
                - "Who directed ..."
                - "What is the title of..."

                The "target_attribute" is what the question should ask.
                {json.dumps(q_dict, indent=4)}""",
                },
            ],
        )

        question1 = response.choices[0].message.content

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""You are to concatenate two questions together, Q1 and Q2. The answer to Q2 is intended to be a clue to help answer Q1, so phrase the whole question for the answer to Q1, using the answer to Q2 as a clue
                    
                    Some guidance:
                    - Phrase this concatenation as a single question. Do not mention the second question or clues. Phrase it naturally.
                    - If Q2 is intended to asks about an actor, and Q1 has a character mentioned, then the Q2 actor plays the Q1 character. Remember to distinguish between characters and actors.
                    - If Q2 is about a genre, then the Film referred to by Q1 belongs to the same genre as what is in Q2. 
                    
                    ** QUESTIONS **
                    Q1: {question1}
                    Q2: {question2}
""",
                },
            ],
        )


        question["question"] = response.choices[0].message.content

    def filter_whole_dataset_duplicates(self, dataset):
        filtered_dataset = []

        for idx, question in enumerate(dataset):
            unique = True
            for other_question in dataset[idx + 1 :]:
                if not unique:
                    break

                if (
                    question["type"] != other_question["type"]
                    or question["answer"] != other_question["answer"]
                    or question["attribute"] != other_question["attribute"]
                ):
                    continue

                if question["type"] == "compare":
                    if (
                        question["compared1"] == other_question["compared1"]
                        and question["compared2"] == other_question["compared2"]
                    ):
                        unique = False
                        continue
                else:  # type is "count"
                    if (
                        question["attribute"] == other_question["attribute"]
                        and question["instance"] == other_question["instance"]
                        and question["answer"] == other_question["answer"]
                    ):
                        unique = False
                        continue

            if unique:
                filtered_dataset.append(question)

        return filtered_dataset

    def save_questions(self, questions, name="questions"):
        with open(f"../generated/{name}.json", "w") as fp:
            json.dump(questions, fp, indent=2)


if __name__ == '__main__':
    e = MoviesDataset(type="INDIRECT", memory_span=8000)
    e.generate_examples(10)
