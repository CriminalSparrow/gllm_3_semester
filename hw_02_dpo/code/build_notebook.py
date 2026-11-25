"""Собрать Colab-ноутбук из компактных шагов, которые вызывают скрипты РК2."""

import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def add(kind, cell_id, source):
        cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """
# РК2 · DPO для консультанта автосервиса

`replay` показывает зафиксированные результаты и проверяет артефакты. `collect` генерирует новые ответы SFT для разметки. `train` обучает адаптер на сохранённых парах и генерирует ответы на тесте. Для `collect` и `train` выберите T4 в меню «Среда выполнения». Слепую разметку выполняет независимый судья, подробности в README и отчёте.
""")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    add("code", "install", "%pip install -q " + " ".join(shlex.quote(item) for item in requirements if item))
    add("code", "settings", """
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from IPython.display import Image, display

MODE = "replay"  # "replay", "collect" или "train"
REPOSITORY = "https://github.com/CriminalSparrow/gllm_3_semester.git"
BRANCH = "egor.sokolov/rk2.DPO"
PROJECT = Path.cwd().resolve()
if PROJECT.name == "hw_02_dpo":
    PROJECT = PROJECT.parent
if not (PROJECT / "hw_02_dpo/code/rank.py").is_file():
    PROJECT = Path("/content/gllm_rk2")
    if not PROJECT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(PROJECT)],
                       check=True)

ROOT = PROJECT / "hw_02_dpo"
OUT = ROOT if MODE == "replay" else ROOT / "review_run" / MODE
OUT.mkdir(parents=True, exist_ok=True)

def run(script, *args):
    subprocess.run([sys.executable, "-u", str(ROOT / "code" / script), *map(str, args)],
                   cwd=PROJECT, check=True)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

assert MODE in {"replay", "collect", "train"}
assert (PROJECT / "hw_01_sft/adapter/adapter_model.safetensors").is_file()
print("Режим", MODE)
if MODE in {"collect", "train"}:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
""")
    add("markdown", "preferences-title", "## Кандидаты и предпочтения")
    add("code", "preferences", """
if MODE == "collect":
    run("collect.py", "--output", OUT, "--batch-size", 16, "--guided-batch-size", 32)
    run("rank.py", "prepare", "--generation", OUT / "logs/preference_generation.json",
        "--package", OUT / "blind_preferences.jsonl", "--mapping", OUT / "preference_mapping.json")
    print("Слепые задания готовы. После разметки выполните rank.py assemble по README.")
else:
    paths = {split: str(ROOT / "data" / f"preference_{split}.jsonl") for split in ("train", "validation")}
    if all(Path(path).is_file() for path in paths.values()):
        preference = load_dataset("json", data_files=paths)
        print({split: len(table) for split, table in preference.items()})
        example = preference["train"][0]
        print(example["prompt"][-1]["content"])
        for label in ("chosen", "rejected"):
            print(f"\\n{label}:\\n{example[label][0]['content']}")
    else:
        print("Preference-датасет появится после генерации и разметки.")
""")
    add("markdown", "dpo-title", "## DPOTrainer и подбор гиперпараметров")
    add("code", "training", """
if MODE == "train":
    run("train.py", "--preference-dir", ROOT / "data", "--output", OUT)
    run("plot.py", "--output", OUT, "--stage", "training")
training_log = OUT / "logs/training.json"
if training_log.is_file():
    training = read_json(training_log)
    print("Среда", training["settings"]["environment"])
    for trial in training["trials"].values():
        print(trial["config"], "валидация", {key: trial["after"][key] for key in (
            "eval_loss", "eval_rewards/accuracies", "eval_rewards/margins")})
    assert sha256(OUT / "adapter/adapter_model.safetensors") == training["selected"]["adapter_sha256"]
    for split, expected in training["settings"]["data_sha256"].items():
        assert sha256(ROOT / "data" / f"preference_{split}.jsonl") == expected
    print("Выбранная конфигурация", training["selected"])
    for name in ("training.png", "rewards.png"):
        display(Image(filename=str(OUT / "plots" / name)))
else:
    print("Журнал обучения появится после запуска DPOTrainer.")
""")
    add("markdown", "eval-title", "## Слепое сравнение SFT и DPO")
    add("code", "evaluation", """
if MODE == "train":
    run("predict.py", "--output", OUT, "--batch-size", 12)
    run("evaluate.py", "prepare", "--output", OUT)
    print("Слепой пакет подготовлен. После разметки выполните evaluate.py aggregate по README.")
summary_path = OUT / "metrics/summary.json"
if summary_path.is_file():
    summary = read_json(summary_path)
    evaluation = read_json(OUT / "logs/evaluation.json")
    for stage, expected in evaluation["mapping"]["answer_sha256"].items():
        assert sha256(OUT / "metrics" / f"{stage}_answers.jsonl") == expected
    for key, section in summary.items():
        print(key, section["wins"], "DPO win rate", section["dpo_win_rate"], "CI", section["dpo_win_rate_ci95"])
    for axis, results in summary["overall"]["axes"].items():
        print(axis, results)
    display(Image(filename=str(OUT / "plots/comparison.png")))
else:
    print("Сводка появится после слепой разметки ответов.")
""")
    add("code", "examples", """
if all((OUT / "metrics" / f"{stage}_answers.jsonl").is_file() for stage in ("sft", "dpo")):
    answers = {stage: {row["id"]: row for row in read_jsonl(OUT / "metrics" / f"{stage}_answers.jsonl")}
               for stage in ("sft", "dpo")}
    scores = OUT / "metrics/judge_scores.csv"
    if scores.is_file():
        with scores.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        ids = [next((row["id"] for row in rows if row["winner"] == winner), None)
               for winner in ("dpo", "sft", "tie")]
    else:
        ids = ["test_03", "test_11"]
    for item_id in filter(None, ids):
        print(answers["sft"][item_id]["question"])
        for stage in ("sft", "dpo"):
            print(f"\\n{stage.upper()}:\\n{answers[stage][item_id]['answer']}")
        print()
""")
    notebook = {"cells": cells, "metadata": {"colab": {"gpuType": "T4", "provenance": []},
                 "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "autoservice_dpo.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
