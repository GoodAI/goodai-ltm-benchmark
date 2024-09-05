import re
import json
from copy import deepcopy
from typing import Callable
from utils.llm import make_system_message, make_user_message, make_assistant_message, LLMContext
from utils.text import selection_range_to_indices, index_context_lines, stamp_content
from datetime import datetime


BASE_SYSTEM_PROMPT = (
    "You are an AI assistant with long-term memory. Your context is not complete, but includes mostly messages from "
    "past interactions that are related to your current situation."
)


def distill_memories(
    user_message: str, memories_ctx: LLMContext, reply: Callable[[LLMContext, float], str],
) -> tuple[str, LLMContext]:

    system_prompt = [make_system_message(BASE_SYSTEM_PROMPT)]
    now = datetime.now()
    prompt_ctx = [make_user_message(stamp_content(
        f"{user_message}\n\n**WAIT!** Do not respond to this message, but annotate relevant information that will help "
        f"you later respond to it. Focus on anything that might not be obvious or is subtle. Do not quickly jump into "
        f"conclusions, and don't annotate anything that is not supported in the text.",
        now, dt=now,
    ))]
    memory_notes = reply(system_prompt + memories_ctx + prompt_ctx, 0)

    return memory_notes, memories_ctx

    sel_prompt_ctx = prompt_ctx + [
        make_assistant_message(memory_notes),
        make_user_message(
            'What information would you like to use in your response? Respond with a JSON string in this format: '
            '"1-3, 5, 8-12"'
        ),
    ]

    numbered_memories_ctx = index_context_lines(memories_ctx)

    # Make N calls with high temperature
    selections = dict()
    for _ in range(3):
        selection_str = reply(system_prompt + numbered_memories_ctx + sel_prompt_ctx, 1)  # TODO: Potential input overflow
        try:
            i = selection_str.find('"')
            j = selection_str.find('"', i + 1)
            assert 0 <= i < j
            selection = selection_range_to_indices(selection_str[i + 1:j])
        except:
            selection = []
        for i in selection:
            selections[i] = selections.get(i, 0) + 1
    selection = sorted(i for i, c in selections.items() if c >= 2)

    selected_memories = extract_memory_lines(numbered_memories_ctx, selection)
    # memory_notes = reply(system_prompt + selected_memories + prompt_ctx, 0)
    return memory_notes, selected_memories


def take_notes_old(
    user_message: str, memories_ctx: LLMContext, reply: Callable[[LLMContext, float], str],
) -> tuple[str, LLMContext]:

    system_prompt = [make_system_message(BASE_SYSTEM_PROMPT)]
    now = datetime.now()
    prompt_ctx = [make_user_message(stamp_content(
        f"Write a short note to your future self that helps you respond to the following message:\n\n"
        f"```text\n{user_message}\n```\n\n"
        f"You will have all the information at hand, so you better focus on making comments, observations and deductions.",
        now, dt=now,
    ))]
    memory_notes = reply(system_prompt + memories_ctx + prompt_ctx, 0)

    return memory_notes, memories_ctx


def discard_irrelevant_memories(
    user_message: str, memories_ctx: LLMContext, thr_interactions: int, reply: Callable[[LLMContext, float], str],
) -> LLMContext:
    # Index interactions and have the agent say what's not relevant
    system_prompt = [make_system_message(BASE_SYSTEM_PROMPT)]
    while True:

        # Copy context and index user messages
        i = 0
        indexed_memories = deepcopy(memories_ctx)
        for m in indexed_memories:
            if m["role"] == "user":
                i += 1
                m["content"] = f"[{i}]" + m["content"]

        # Ask the agent to mark irrelevant content
        now = datetime.now()
        prompt_ctx = [make_user_message(stamp_content(
            "You've now received the following message:\n\n"
            f"```text\n{user_message}\n```\n\n"
            "But wait, do not respond to the message just yet.\n"
            "I want you to tell me if there are irrelevant messages.\n"
            "Respond only with a JSON list of integers.",
            now, dt=now,
        ))]
        memories_sel_str = reply(system_prompt + indexed_memories + prompt_ctx, 0).replace(" ", "")
        m = re.match(r"\d+(-\d+)?(,\d+(-\d+)?)*", memories_sel_str)
        indices = [] if m is None else selection_range_to_indices(m.string)
        memories_sel = []
        i = 0
        for j, m in enumerate(memories_ctx):
            if m["role"] == "user":
                i += 1
                if i not in indices:
                    memories_sel.extend(memories_ctx[j:j+2])

        memories_ctx = memories_sel
        if len(indices) == 0 or len(memories_ctx) <= thr_interactions:
            return memories_ctx


def message_notes_and_analysis(
    user_message: str, memories_ctx: LLMContext, now: datetime, reply: Callable[[LLMContext, float], str]
) -> str:
    system_prompt = [make_system_message(BASE_SYSTEM_PROMPT)]
    context = system_prompt + deepcopy(memories_ctx) + [make_user_message(stamp_content(user_message, now, dt=now))]
    user_messages = [m for m in context[:-1] if m["role"] == "user"]
    notes = dict()
    for msg_idx, msg in enumerate(user_messages):
        ctx = deepcopy(context)
        indented_msg = "(From User)\n" + "\n".join("    " + line for line in msg["content"].splitlines())
        ctx[-1]["content"] += (
            "\n\n**WAIT!** Do not respond to this message, but do some analysis first. Let's focus on one of your past "
            f"interactions:\n\n{indented_msg}\n\nIs it relevant to the current situation? Does it contain key "
            "information? Respond with a list of annotations, or with an empty list if there is nothing to say. Do not "
            "speculate or write anything that is not explicitly supported by the text. Format your response as a JSON "
            "list of strings e.g. [\"1st annotation\", \"2nd annotation\", ...]"
        )
        response = reply(ctx, 0)
        i = response.find("[")
        j = response.find("]", i + 1)
        try:
            notes[msg_idx + 1] = json.loads(response[i:j+1])
        except:
            pass

    if len(notes) > 0:
        notes_txt = list()
        for msg_idx, msg_notes in notes.items():
            notes_txt.append(f"Message {msg_idx}:")
            notes_txt.extend("- " + n for n in msg_notes)
        notes_txt = "\n".join(notes_txt)
        ctx = index_user_messages(context)
        ctx[-1]["content"] += (
            "\n\n**WAIT!** Do not respond to this message yet, but do some analysis first. Take a look at these notes:\n\n"
            f"{notes_txt}\n\nWhat do you think? Write a global comment that will help you generate a spot-on response."
        )
        meta_notes = reply(ctx, 0)
        context = index_user_messages(context)
        context[0]["content"] += (
            f"\n\nThese are notes that you have extracted from your previous interactions:\n{notes}\n\nAnd this is an "
            f"analysis that you performed on the current input:\n\n```text\n{meta_notes}\n```"
        )

    return reply(context, 0)


def index_user_messages(context: LLMContext, template: str = "[{idx}]{content}") -> LLMContext:
    context = deepcopy(context)
    i = 0
    for msg in context:
        if msg["role"] == "user":
            i += 1
            msg["content"] = template.format(idx=i, content=msg["content"])
    return context


def extract_memory_lines(numbered_memories_ctx: LLMContext, indices: list[int]) -> LLMContext:
    selection = set(indices)
    selected_memories = list()
    i = 0
    for m in numbered_memories_ctx:
        lines = list()
        msg_lines = m["content"].splitlines()
        last_idx_added = -1
        for j, l in enumerate(msg_lines):
            if i in selection:
                if j - last_idx_added > 1:
                    lines.append("...")
                k = l.find("] ")
                if k >= 0:
                    l = l[k + 2:]
                lines.append(l)
                selection.remove(i)
                last_idx_added = j
            i += 1
        if 0 <= last_idx_added < len(msg_lines) - 1:
            lines.append("...")
        if len(lines) > 0:
            selected_memories.append(dict(role=m["role"], content="\n".join(lines)))
    return selected_memories


def reply_from_distilled(user_message: str, notes: str, memories: LLMContext, reply: Callable[[LLMContext], str]) -> str:
    system_prompt = [make_system_message(
        f"{BASE_SYSTEM_PROMPT}\n\nJust after receiving the latest message, you took the following notes, with the "
        f"intention of helping you focus on the right information and produce a better response:\n\n```text\n{notes}\n```"
    )]
    now = datetime.now()
    user_prompt = [make_user_message(stamp_content(user_message, now, dt=now))]
    return reply(system_prompt + memories + user_prompt)
