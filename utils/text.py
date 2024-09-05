import string
from rouge_score import rouge_scorer
from datetime import datetime, timedelta
from utils.llm import LLMContext


def rouge_l(reference: str, candidate: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)["rougeL"].fmeasure


def standardise(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table)


def truncate(s: str, length: int = 40) -> str:
    if len(s) <= length:
        return s
    return s[:length - 3].rstrip(string.punctuation) + "..."


def td_format(td: timedelta) -> str:
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


def stamp_content(content: str, now: datetime, ts: float = None, dt: datetime = None) -> str:
    assert ts is not None or dt is not None
    if ts is not None:
        dt = datetime.fromtimestamp(ts)
    return f"(Received on {str(dt)[:-7]}, which is {td_format(now - dt)})\n{content}"


def index_lines(text: str, max_line_size: int = 100) -> list[str]:
    lines = list()
    i = 0
    for line in text.strip().splitlines():
        if line.strip() == "":
            lines.append(line)
            i += 1
            continue
        j = 0
        while True:
            chunk = line[j:j + max_line_size]
            if chunk == "":
                break
            chunk = f"[{i:3d}] {chunk}"
            lines.append(chunk)
            i += 1
            j += max_line_size
    return lines


def index_context_lines(context: LLMContext, max_line_size: int = 100) -> LLMContext:
    numbered_ctx = [
        {"role": m["role"], "content": index_lines(m["content"], max_line_size=max_line_size)}
        for m in context
    ]
    i = 0
    for m in numbered_ctx:
        content_lines = m["content"]
        for j in range(len(content_lines)):
            k = content_lines[j].find("] ")
            if k >= 0:
                content_lines[j] = f"[{i:3d}" + content_lines[j][k:]
            i += 1
    return [{"role": m["role"], "content": "\n".join(m["content"])} for m in numbered_ctx]


def selection_range_to_indices(selection: str) -> list[int]:
    indices = set()
    for sel in selection.split(","):
        sel = sel.strip()
        if sel == "":
            continue
        if "-" in sel:
            a, b = [int(part.strip()) for part in sel.split("-")]
            indices.update(range(a, b + 1))
        else:
            indices.add(int(sel.strip()))
    return sorted(indices)
