import os
import json
from typing import List, Tuple

from utils.llm import ask_llm, make_system_message, make_user_message
from utils.data import get_file
from dataclasses import dataclass
from dataset_interfaces.interface import DatasetInterface, TestExample

MSC_DATASETS_VERSION = "v0.1"
MSC_URL = f"http://parl.ai/downloads/msc/msc_{MSC_DATASETS_VERSION}.tar.gz"
MSC_FILENAME = f"msc_{MSC_DATASETS_VERSION}.tar.gz"
MSC_CHECKSUM = "e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60"
QUESTION_TEMPLATE = """
Pay attention to the following conversation between {speaker_1} and {speaker_2}:

{conversation}

Complete the conversation: what will {speaker_2} answer?
""".strip()
EVAL_SYSTEM_TEMPLATE = """
Your name is {name} and you are characterised by the following statements:
{persona}
"""
EVAL_TEMPLATE = """
In the context of this conversation:

{conversation}

Do you think you would continue it with the following line?
{answer}

Respond only with a JSON in the form
{{
  "thoughts": "your thoughts about the suggested response and how it relates to your persona",
  "reasoning": "explain why do you think it is / isn't an answer you would give",
  "conclusion": "yes" or "no",
}}
""".strip()


def make_path_tree(d: dict, path: str) -> dict:
    for part in path.split("/"):
        if part not in d:
            d[part] = dict()
        d = d[part]
    return d


def dialog_content(dialog: list[dict[str, str]]) -> str:
    return "\n".join(m["text"] for m in dialog)


def dialog_as_task(dialog: list[dict[str, str]]) -> dict[str, str]:
    messages = [f"{m['id']}: {m['text']}" for m in dialog[:-1]]
    return dict(
        dialog="\n".join(messages),
        last_speaker=dialog[-1]["id"],
        other_speaker=dialog[-2]["id"],
        gold_answer=dialog[-1]["text"],
    )


def fix_ids(dialog: list[dict[str, str]]):
    switch_ids = {"Speaker 1": "Speaker 2", "Speaker 2": "Speaker 1"}
    for i, msg in enumerate(dialog):
        if msg["id"] in switch_ids:
            continue
        j = i - 1 if i > 0 else i + 1
        msg["id"] = switch_ids[dialog[j]["id"]]


def read_tar_file(dataset_name: str) -> dict:
    import json
    import tarfile

    tar_path = get_file(dataset_name, MSC_URL, MSC_FILENAME, checksum=MSC_CHECKSUM)
    tar_dict = dict()
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if not (m.isfile() and m.name.endswith(".txt")):
                continue
            content = [json.loads(line) for line in tar.extractfile(m).readlines()]
            tree_path, name = os.path.split(m.name)
            name = name.removesuffix(".txt")
            make_path_tree(tar_dict, tree_path)[name] = content
    return tar_dict


def reconstruct_test_chats(tar_dict: dict) -> dict:
    """
    :param tar_dict: Raw data loaded from tar file.
    :return: Chats by chat_id. A chat is a list of sessions.
    Every session is a dict containing the following data:
    - "dialog"
    - "personas"
    - "time_num" Number of time_unit from previous session.
    - "time_unit"
    - "time_back" Time elapsed wrt. most recent session.
    """
    chats_dict = dict()

    # Initial conversations are in msc_personasummary
    to_speaker = {"bot_0": "Speaker 1", "bot_1": "Speaker 2"}
    msc_summary = tar_dict["msc"]["msc_personasummary"]
    for chat in msc_summary["session_1"]["test"]:
        dialog = [
            {"id": to_speaker[m["id"]], "text": m["text"]} for m in chat["dialog"]
        ]
        chats_dict[chat["initial_data_id"]] = [
            dict(
                dialog=dialog,
                personas=chat["init_personachat"]["init_personas"],
            )
        ]

    # Following conversations are in msc_dialogue
    msc_dialogue = tar_dict["msc"]["msc_dialogue"]
    for session_nr in range(2, 6):
        for chat in msc_dialogue[f"session_{session_nr}"]["test"]:
            dialog = [{"id": m["id"], "text": m["text"]} for m in chat["dialog"]]
            fix_ids(dialog)
            chat_id = chat["metadata"]["initial_data_id"]
            chats_dict[chat_id].append({"dialog": dialog, "personas": chat["personas"]})

    # Time information
    time_keys = ["time_num", "time_unit", "time_back"]
    for chat in msc_dialogue["session_5"]["test"]:
        chat_id = chat["metadata"]["initial_data_id"]
        for i, dialog in enumerate(chat["previous_dialogs"]):
            chats_dict[chat_id][i].update({k: dialog[k] for k in time_keys})
        last_chat = chats_dict[chat_id][-1]
        last_chat["time_num"] = 0
        last_chat["time_unit"] = "seconds"
        last_chat["time_back"] = "now"

    return chats_dict


@dataclass
class MultiSessionChatDataset(DatasetInterface):
    name: str = "Multi-Session Chat"
    description: str = "Long conversation records that span across days."

    def get_chats(self) -> list[list[dict]]:
        tar_dict = read_tar_file(self.name)
        chats_dict = reconstruct_test_chats(tar_dict)
        chats = [chats_dict[k] for k in sorted(chats_dict.keys())]
        self.random.shuffle(chats)
        return chats

    def generate_examples(self, num_examples: int) -> list[TestExample]:
        test_examples = list()
        for chat in self.get_chats()[:num_examples]:
            full_dialog = [msg for sess in chat for msg in sess["dialog"]]
            task = dialog_as_task(full_dialog)
            question = QUESTION_TEMPLATE.format(
                speaker_1=task["other_speaker"],
                speaker_2=task["last_speaker"],
                conversation=task["dialog"],
            )
            last_dialog_lines = "\n".join(task["dialog"].splitlines()[-10:])
            answer_data = dict(
                name=task["last_speaker"],
                persona="\n".join(f"- {p}" for p in chat[-1]["personas"][1]),
                gold_answer=task["gold_answer"],
                conversation=last_dialog_lines,
            )
            script = [question]
            is_question = [True]

            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=[json.dumps(answer_data)],
                is_question=is_question,
            )
            test_examples.append(example)
        return test_examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        # TODO: Consider computing the ROUGE-L score
        # pip install rouge-score
        answer_data = json.loads(expected_answers[0])
        context = [
            make_system_message(
                EVAL_SYSTEM_TEMPLATE.format(
                    name=answer_data["name"], persona=answer_data["persona"]
                )
            ),
            make_user_message(
                EVAL_TEMPLATE.format(
                    conversation=answer_data["conversation"],
                    answer=answer_data["gold_answer"],
                )
            ),
        ]
        response = ask_llm(context, "gpt-4", temperature=0)
        try:
            json_dict = json.loads(response)
            score = 1 if "yes" == json_dict["conclusion"].lower().strip() else 0
            return score, 1, [json_dict["reasoning"]]
        except (json.JSONDecodeError, KeyError) as exc:
            return 0, 1, [str(exc)]

