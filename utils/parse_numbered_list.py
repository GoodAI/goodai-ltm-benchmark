import json
from utils.constants import DATA_DIR

FOLDER = DATA_DIR.joinpath("input_raw_data/movie_qa")
genres = [
    "horror",
    "scifi",
    "romance",
    "adventure",
    "children",
    "drama",
    "thriller",
    "fantasy",
    "comedy",
    "mystery",
]


def main():
    for g in genres:
        fname_read = FOLDER.joinpath(f"{g}_numbered.txt")
        name_list = []
        with open(fname_read, "r") as f:
            names = f.readlines()
            for n in names:
                name_split = n.split(" ")
                name_list.append(" ".join(name_split[1:]))

        fname_write = FOLDER.joinpath(f"{g}_list.json")
        with open(fname_write, "w") as f:
            json.dump(name_list, f)


if __name__ == "__main__":
    main()
