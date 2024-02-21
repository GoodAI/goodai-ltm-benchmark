import enum
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent
DATA_DIR = MAIN_DIR.joinpath("data")
TESTS_DIR = DATA_DIR.joinpath("tests")
PERSISTENCE_DIR = DATA_DIR.joinpath("persistence")
REPORT_TEMPLATES_DIR = MAIN_DIR.joinpath("reporting/templates")
REPORT_OUTPUT_DIR = DATA_DIR.joinpath("reports")
GOODAI_GREEN = (126, 188, 66)
GOODAI_RED = (188, 66, 66)


class EventType(enum.Enum):
    SEND_MESSAGE = 0
    BEGIN = 1
    END = 2
    SEND_FILL = 3
    RESPONSE_MESSAGE = 4
    RESPONSE_FILL = 5
    WAIT = 6


EVENT_SENDER = {
    EventType.SEND_MESSAGE: "Test",
    EventType.SEND_FILL: "System",
    EventType.RESPONSE_MESSAGE: "Agent",
    EventType.RESPONSE_FILL: "Agent",
}


METRIC_ALT = dict(
    accuracy=(
        "(Average Test Accuracy - %) The accuracy can be computed for each test by dividing the score achieved by the "
        "maximum score possible for that test. We then average all tests' accuracies together. This can be viewed "
        "as a uniformly-weighted score average."
    ),
    speed=(
        "(Average Tests per Hour) How many tests, on average, the agent completes within an hour of running the "
        "benchmark."
    ),
    cost=(
        "(USD) Overall cost of running the agent for the whole benchmark, in US dollars. We use the negative value to "
        "obtain the Economy."
    ),
    verbosity=(
        "(Tokens) The number of tokens that comprise the complete benchmark log. The longer the agent's responses are, "
        "the larger this metric is. The negative of this value is the Conciseness."
    ),
    score=(
        "(Points) Each test gives a different amount of score points. This score is the result of adding up the "
        "number of score points achieved in all tests."
    ),
    ltm=(
        "(LTM Points) The GoodAI LTM Score is computed from the number of tokens that separate the relevant information "
        "from the final question in each test. In every case, this distance is weighted by the accuracy achieved in the "
        "corresponding test."
    ),
)
METRIC_NAMES = {key: "LTM Score" if key == "ltm" else key.capitalize() for key in METRIC_ALT.keys()}
METRIC_UNITS = dict(
    accuracy="%",
    speed="tests per hour",
    cost="USD",
    verbosity="tokens",
    score="points",
    ltm="LTM points",
)
SPIDER_LABELS_OVERRIDE = dict(
    cost="Economy",
    verbosity="Conciseness",
)
