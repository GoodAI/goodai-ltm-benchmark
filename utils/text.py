import tiktoken
from rouge_score import rouge_scorer


def token_len(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def rouge_l(reference: str, candidate: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)["rougeL"].fmeasure
