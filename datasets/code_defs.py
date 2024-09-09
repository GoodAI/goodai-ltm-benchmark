from dataclasses import dataclass
from typing import List, Any, Tuple

from goodai.helpers.json_helper import sanitize_and_parse_json

from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.llm import make_user_message

code_pairs = [
    ("""def function() -> dict:
    results = dict()
    csv_path = input("Enter the path to the CSV results: ")
    with open(csv_path) as fd:
        for line in fd:
            if line[:2] in ["Is", "1k", "32", "12", "20", "50"]:
                k = line.split(" ")[0]
                if k == "1k":
                    k = "2k"
                results[k] = dict()
            elif line[:3] in ["gpt", "Cla", "LTM", "Mix", "Met", "Gem"]:
                model, _, score, std = line.split(",")[:4]
                score = score.strip()
                std = std.strip()
                score = float(score) if score != "" else 0
                std = float(std) if std != "" else 0
                model = aliases[model]
                results[k][model] = dict(score=score, score_std=std)
            if line.startswith("Others not included"):
                break
    return results
""",
    "loads csv data from a file"),

    ("""def function(td: datetime.timedelta) -> str:
    seconds = int(td.total_seconds())
    periods = [
        ('year', 3600*24*365), ('month', 3600*24*30), ('day', 3600*24), ('hour', 3600), ('minute', 60), ('second', 1)
    ]
    parts = list()
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            parts.append("%s %s%s" % (period_value, period_name, has_s))
    if len(parts) == 0:
        return "just now"
    if len(parts) == 1:
        return f"{parts[0]} ago"
    return " and ".join([", ".join(parts[:-1])] + parts[-1:]) + " ago"
""",
    "formats absolute timestamps into relative"
    ),

    ("""def function(self, num_examples):
    examples = []

    for _ in range(num_examples):
        is_question = []
        colours = []
        script = []
        renderer = pystache.Renderer()

        for change in range(self.colour_changes):
            colour = self.random.choice(COLOURS)
            if colour == "None":
                name_stmt = "I have no favourite colour."
            else:
                name_stmt = renderer.render(self.random.choice(STATEMENTS), {"colour": colour})
            colours.append(colour)
            script.append(name_stmt)
            is_question.append(False)

        script.append(self.question)
        is_question.append(True)
        answer_list = [colours[-1]]
        example = TestExample(
            dataset_generator=self,
            script=script,
            expected_responses=answer_list,
            is_question=is_question,
        )
        examples.append(example)

    return examples""",
     "generates a list of examples for a test"
     ),

    ("""def function(self, user_content: str, agent_response: Optional[str] = None) -> str:
    context = self.build_llm_context(user_content)
    response = self.completion(context, temperature=self.llm_temperature, label="reply")
    user_message = Message(role='user', content=user_content, timestamp=self.current_time)
    self.message_history.append(user_message)
    assistant_message = Message(role='assistant', content=response, timestamp=self.current_time)
    self.message_history.append(assistant_message)
    return response
""",
    "constructs a reply to a user based on content of their message"
    ),

    ("""def function(a: TestExample, b: TestExample, incompatibilities: list[set[type]]) -> bool:
    cls_a, cls_b = type(a.dataset_generator), type(b.dataset_generator)
    if cls_a is cls_b:
        return False
    for inc_set in incompatibilities:
        if cls_a in inc_set and cls_b in inc_set:
            return False
    return True
""",
     "determines if two examples of tests are incompatible with each other"
     ),

    ("""def function(name: str, max_prompt_size: Optional[int], run_name: str, is_local=False) -> ChatSession:
    kwargs = {"max_prompt_size": max_prompt_size} if max_prompt_size is not None else {}
    kwargs["run_name"] = run_name
    kwargs["is_local"] = is_local

    if name == "gemini":
        return GeminiProInterface(run_name=run_name)
    if (match := re.match(r"^fifo\((?P<file>.+)\)$", name)) is not None:
        return FifoAgentInterface(fifo_file=Path(match.groupdict()["file"]), **kwargs)
    if name.startswith("ltm_agent"):
        match = re.match(r"^ltm_agent\((?P<model>.+)\)$", name)
        if match is None:
            raise ValueError(f"Unrecognized LTM Agent {repr(name)}.")
        return LTMAgentWrapper(model=match.groupdict()["model"], **kwargs)
    if name == "length_bias":
        return LengthBiasAgent(model=GPT_4_TURBO_BEST, **kwargs)
    if name.startswith("cost("):
        in_cost, out_cost = [float(p.strip()) / 1_000 for p in name.removeprefix("cost(").removesuffix(")").split(",")]
        return CostEstimationChatSession(cost_in_token=in_cost, cost_out_token=out_cost, **kwargs)
    if name == "human":
        return HumanChatSession(**kwargs)
    if name.startswith("huggingface/"):
        kwargs.pop("is_local")
        return HFChatSession(model=name, **kwargs)
    if name == "memory_bank":
        return MemoryBankInterface(api_url="http://localhost:5000")
    if name.startswith("memgpt"):
        match = re.match(r"^memgpt\((?P<model>.+)\)$", name)
        if match is None:
            raise ValueError(f"Unrecognized MemGPT Agent {repr(name)}.")
        return MemGPTInterface(model=match.groupdict()["model"], **kwargs)
""",
     "instantiates one of a number of different LLM based agents"),

    ("""def function(self, questions: List[str], responses: List[str], expected_answers: List[str]) -> Tuple[int, int, List[str]]:
    if expected_answers[0] in responses[0]:
        max_score = 1
        score = 1
        reasons = ["The correct joke is recounted."]
        return score, max_score, reasons
    else:
        return self.evaluate_correct_gpt(questions, responses, expected_answers)
""",
    "evaluating whether a given answer is correct when compared to one which is expected"),

    ("""def function(self):
    self.expected_responses.append("The agent notices the mix-up.")
    context = [
        make_system_message(notice_mishap_prompt),
        make_user_message(f"Customer: {self.action.reply}"),
    ]
    noticed = self.gpt_bool_check(context, "noticed")
    if not noticed:
        self.reasoning.append("The agent does not notice the mishap.")
        raise RestaurantOrderFailed
    self.reasoning.append("The agent notices the unordered meal.")
    self.score += 1
""",
     "evaluates whether an agent will notice a mistake in a meal order"
     ),

    ("""def function(arr):
for n in range(len(arr) - 1, 0, -1):

    for i in range(n):
        if arr[i] > arr[i + 1]:

            swapped = True
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
""",
     "performs a bubble sort"
     ),

    ("""import random
 
def function(a):
    n = len(a)
    while (function1(a)== False):
        function2(a)
 
def function1(a):
    n = len(a)
    for i in range(0, n-1):
        if (a[i] > a[i+1] ):
            return False
    return True
 
def function2(a):
    n = len(a)
    for i in range (0,n):
        r = random.randint(0,n-1)
        a[i], a[r] = a[r], a[i]
""",
    "performs a bogosort by randomly permuting the array and checking to see if it is sorted"
     )

]



@dataclass
class CodeDefinitionsDataset(DatasetInterface):

    name: str = "CodeDefinitions"
    description: str = "Shows the agent a number of anonymised functions and asks the agent to retrieve function by descriptions of their operation."
    total_functions: int = 4
    target_functions: int =2

    def __post_init__(self):
        assert self.total_functions >= self.target_functions, "There needs to be at least as many functions to ask questions about as questions"
        assert self.total_functions <= len(code_pairs), "There are too few functions to ask questions about"

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        examples = []

        for _ in range(num_examples):
            functions_in_example = self.random.sample(code_pairs, self.total_functions)
            target_functions_in_example = self.random.sample(functions_in_example, self.target_functions)

            script = ["I am going to give you a bunch of functions for you to remember for me. No need to comment on them, just be able to remember then when I ask. Okay?"]
            expected_responses=[]
            is_question = [False]

            for func_pair in functions_in_example:
                script.append(f"Here is a function:\n ```python\n{func_pair[0]}\n```\nDon't comment on the function, I already know what it does.")
                is_question.append(False)

            question_str = "Please recall for me, the exact code for the function that {operation}."

            for func_pair in target_functions_in_example:
                script.append(question_str.format(operation=func_pair[1]))
                expected_responses.append(func_pair[0])
                is_question.append(True)

            examples.append(TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=expected_responses,
                is_question=is_question,
            ))

        return examples


    def evaluate_correct(self, questions: List[str], responses: List[str], expected_answers: List[Any]) -> Tuple[int, int, List[str]]:

        prompt = """Your job is to determine whether the retrieved code matches the expected code. Ignore formatting changes and just focus on their functionality.
**************
RETRIEVED:
```
{retrieved}
``` 
**************
EXPECTED:
```
{expected}
```

Respond in JSON like:

{{
    "matches": bool // Whether the retrieved and expected code match. 
}}"""

        score = 0
        reasoning = []

        for q, r, e in zip(questions, responses, expected_answers):

            # If there is a string match then score it and move on
            if r == e:
                score += 1
                reasoning.append("Correct code recalled (string match)")
                continue

            context = [make_user_message(prompt.format(retrieved=r, expected=e))]
            judgement_str = self.ask_llm(context, model="gpt-4o-mini")
            judgement_json = sanitize_and_parse_json(judgement_str)

            if judgement_json["matches"]:
                score += 1
                reasoning.append("Correct code recalled (llm check)")
            else:
                reasoning.append("Incorrect code recalled")

        return score/len(expected_answers), 1, reasoning



