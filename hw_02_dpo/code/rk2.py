"""Данные и общие настройки двух этапов эксперимента."""

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT.parent / "hw_01_sft"
sys.path.insert(0, str(SFT / "code"))
from common import SYSTEM, check_answer, read_jsonl, write_jsonl

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SFT_CONFIG = json.loads((SFT / "config.json").read_text(encoding="utf-8"))
AXES = ("safety", "correctness", "usefulness")
RISK_TOPICS = {
    "Красная лампа давления масла",
    "Мигает Check Engine и двигатель троит",
    "Мягкая педаль тормоза или утечка тормозной жидкости",
    "Перегрев двигателя и утечка антифриза",
}
LEGACY_RISK_IDS = {f"test_{n:02d}" for n in (2, 3, 4, 5, 14, 15, 16, 17)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normal(text):
    return re.sub(r"[^\w]+", " ", text.casefold().replace("ё", "е")).strip()


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def legacy_preference_prompts():
    rng = random.Random(CONFIG["seed"])
    result = []
    for split in ("train", "validation"):
        by_topic = defaultdict(list)
        for row in read_jsonl(SFT / f"data/{split}.jsonl"):
            by_topic[row["topic"]].append(row)
        for topic, rows in sorted(by_topic.items()):
            if split == "train":
                rows = rng.sample(rows, k=CONFIG["train_prompts_per_topic"])
            for row in sorted(rows, key=lambda row: row["id"]):
                _, user, assistant = row["messages"]
                result.append({"id": row["id"], "split": split, "topic": topic,
                               "risk": topic in RISK_TOPICS, "question": user["content"],
                               "reference": assistant["content"]})
    return result


def guided_prompts(paths=None):
    sources = paths if paths is not None else sorted((ROOT / "data").glob("prompt_pool_*.jsonl"))
    return [row for source in sources for row in read_jsonl(source)]


def preference_prompts():
    return legacy_preference_prompts() + guided_prompts()


def test_prompts():
    old = [{**row, "suite": "rk1", "risk": row["id"] in LEGACY_RISK_IDS}
           for row in read_jsonl(SFT / "data/test.jsonl")]
    new = [{**row, "suite": "fresh"} for row in read_jsonl(ROOT / "data/test_fresh.jsonl")]
    return old + new


def validate_splits():
    rk1 = [row["messages"][1]["content"] for split in ("train", "validation")
           for row in read_jsonl(SFT / f"data/{split}.jsonl")]
    extension = guided_prompts()
    pool = legacy_preference_prompts() + extension
    if len({r["id"] for r in pool}) != len(pool):
        raise ValueError("ID в пуле preference повторяются")
    if any(r["split"] not in {"train", "validation"} or type(r["risk"]) is not bool for r in extension):
        raise ValueError("Некорректное разбиение или флаг риска")
    prompts = [normal(q) for q in rk1] + [normal(row["question"]) for row in test_prompts() + extension]
    if len(prompts) != len(set(prompts)):
        raise ValueError("Один и тот же нормализованный вопрос встретился дважды")
