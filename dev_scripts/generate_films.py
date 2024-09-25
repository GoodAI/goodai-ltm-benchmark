import json
import os
from typing import List, Dict, Tuple

from faker import Faker
import random

from openai import OpenAI

from utils.constants import DATA_DIR

MOVIES_TO_GENERATE = 100
NUM_DIRECTORS = 24
NUM_ACTORS = 60


client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def load_titles(genres_list):
    val = {}
    for g in genres_list:
        with open(DATA_DIR.joinpath(f"movies", "movie_names", f"{g.lower()}_list.json"), "r") as f:
            line = f.readline()
            val[g.lower()] = json.loads(line)

    return val


def generate_names(num_to_generate: int) -> List[Tuple[str, str]]:
    fake = Faker(["en_US", "en_IE", "it_IT", "es", "fr_FR"])
    names_genders = []
    for i in range(num_to_generate):
        gender = random.choice(["male", "female"])
        name = fake.unique.name_male() if gender == "male" else fake.unique.name_female()

        names_genders.append((name, gender))
    return names_genders


def generate_films(directors_list: List[str], actors_list: List[Tuple[str, str]], num_films: int):
    years = list(range(1990, 2023))
    fake = Faker(["en_US", "en_IE", "it_IT", "es", "fr_FR"])

    genres = [
        "Horror",
        "SciFi",
        "Adventure",
        "Children",
        "Drama",
        "Fantasy",
        "Comedy",
        "Thriller",
        "Mystery",
        "Romance",
    ]

    titles = load_titles(genres)
    films = []
    for i in range(num_films):
        # Get a genre and a unique title
        genre = random.choice(genres)
        chosen_title = random.choice(titles[genre.lower()])
        titles[genre.lower()].remove(chosen_title)

        # Get cast + characters
        cast = random.sample(actors_list, random.randint(3, 7))
        characters = []
        for c in cast:
            character_name = fake.unique.name_male() if c[1] == "male" else fake.unique.name_female()
            characters.append(character_name)

        new_film = {
            "title": chosen_title,
            "genre": genre,
            "year": random.choice(years),
            "director": random.choice(directors_list),
            "cast": [x[0] for x in cast],
            "characters": characters,
        }
        films.append(new_film)
    return films


def generate_synopses(films):
    for idx, film in enumerate(films):
        to_show = {"title": film["title"], "characters": film["characters"]}
        print(
            f"{idx}/{MOVIES_TO_GENERATE}. Generating synopsis for:\n{json.dumps(film, indent=2)}"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Create a very short synopsis for this film. The synopsis should be of the plot, not be advertisement copy, and should not mention the title of the film anywhere. Write the synopsis without using the words {film['title']}:\n{json.dumps(to_show, indent=4)}",
                    },
                ],
            )
            film["synopsis"] = response.choices[0].message.content
        except:
            continue


def save_dataset(films: List[Dict[str, str]]):
    with open(DATA_DIR.joinpath("movies", "movies_list.json"), "w") as fp:
        json.dump(films, fp, indent=2)


def main():
    directors = [x[0] for x in generate_names(NUM_DIRECTORS)]
    actors = generate_names(NUM_ACTORS)

    films = generate_films(directors, actors, MOVIES_TO_GENERATE)

    # Generate synopses for these
    generate_synopses(films)

    # Save to file
    save_dataset(films)

    print(json.dumps(films, indent=4))


if __name__ == "__main__":
    main()
