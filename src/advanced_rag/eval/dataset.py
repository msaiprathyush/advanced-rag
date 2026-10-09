import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = "evals/eval_set.jsonl"


@dataclass
class EvalRow:
    id: str
    question: str
    reference_answer: str = ""
    gold_arxiv_ids: list[str] = field(default_factory=list)
    answerable: bool = True


def load_eval_set(path: str = DEFAULT_PATH) -> list[EvalRow]:
    rows = []
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            rows.append(EvalRow(**json.loads(ln)))
    return rows
