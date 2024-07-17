import string
from rouge_score import rouge_scorer


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
