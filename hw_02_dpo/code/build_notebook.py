"""Сборка ноутбука РК2 из ячеек запуска и просмотра результатов."""

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
# РК2. DPO для консультанта автосервиса

Сравниваем модель после SFT с той же моделью после обучения на парах ответов.
По умолчанию ноутбук показывает результаты готового запуска. Настройки повторного запуска приведены ниже.
""")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    add("code", "install", "%pip install -q " + " ".join(shlex.quote(item) for item in requirements if item))
    add("code", "imports", """
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display
""")
    add("markdown", "settings-title", """
## Настройки

- `replay` — посмотреть готовые результаты.
- `collect` — получить ответы SFT для разметки предпочтений.
- `train` — обучить DPO на сохранённых парах и подготовить ответы к слепой оценке.

Для `collect` и `train` нужна T4. Порядок разметки описан в README.
""")
    add("code", "settings", """
MODE = "replay"

project = Path.cwd().resolve()
if project.name == "hw_02_dpo":
    project = project.parent
if not (project / "hw_02_dpo").is_dir():
    project = Path("/content/gllm_rk2")
    if not project.exists():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "egor.sokolov/rk2.DPO",
            "https://github.com/CriminalSparrow/gllm_3_semester.git", str(project),
        ], check=True)

root = project / "hw_02_dpo"
output = root if MODE == "replay" else root / "review_run" / MODE


def run(script, *args):
    subprocess.run([sys.executable, "-u", str(root / "code" / script), *map(str, args)], check=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
""")
    add("markdown", "preferences-title", "## Пары ответов")
    add("code", "preferences", """
if MODE == "collect":
    run("collect.py", "--output", output, "--batch-size", 16, "--guided-batch-size", 32)
    run("rank.py", "prepare", "--generation", output / "logs/preference_generation.json",
        "--package", output / "blind_preferences.jsonl", "--mapping", output / "preference_mapping.json")
else:
    pairs = {
        split: pd.read_json(root / "data" / f"preference_{split}.jsonl", lines=True)
        for split in ("train", "validation")
    }
    display(pd.DataFrame({"Пар": [len(pairs["train"]), len(pairs["validation"])]},
                         index=["Обучение", "Валидация"]))
    example = pairs["train"].set_index("id").loc["ext_n_002"]
    print(example["prompt"][-1]["content"])
    for label in ("chosen", "rejected"):
        print(f"\\n{label}\\n{example[label][0]['content']}")
""")
    add("markdown", "dpo-title", "## Обучение DPO")
    add("code", "training", """
if MODE == "train":
    run("train.py", "--preference-dir", root / "data", "--output", output)
    run("plot.py", "--output", output, "--stage", "training")

if MODE != "collect":
    training = read_json(output / "logs/training.json")
    trial = training["trials"][training["selected"]["id"]]
    config = trial["config"]
    device = training["settings"]["environment"]["device"]
    print(f"{device} · β = {config['beta']} · скорость обучения = {config['learning_rate']:g}")
    fields = {"DPO loss": "eval_loss", "Разница наград": "eval_rewards/margins"}
    values = pd.DataFrame({
        "До": [trial["before"][key] for key in fields.values()],
        "После": [trial["after"][key] for key in fields.values()],
    }, index=list(fields))
    display(values.round(3))
    for name in ("training.png", "rewards.png"):
        display(Image(filename=str(output / "plots" / name)))
""")
    add("markdown", "eval-title", "## Сравнение на тесте")
    add("code", "evaluation", """
if MODE == "train":
    run("predict.py", "--output", output, "--batch-size", 12)
    run("evaluate.py", "prepare", "--output", output)

summary_path = output / "metrics/summary.json"
if MODE == "replay" or summary_path.is_file():
    summary = read_json(summary_path)
    rows = []
    for key, name in [("overall", "Все вопросы"), ("rk1", "РК1"), ("fresh", "Дополнительные")]:
        result = summary[key]
        wins = result["wins"]
        rows.append([name, wins["sft"], wins["dpo"], wins["tie"], f"{result['dpo_win_rate']:.1%}"])
    display(pd.DataFrame(rows, columns=["Тест", "Победы SFT", "Победы DPO", "Ничьи", "Доля побед DPO"]))
    display(Image(filename=str(output / "plots/comparison.png")))
""")
    add("markdown", "examples-title", "## Примеры ответов")
    add("code", "examples", """
if MODE != "collect":
    answers = {
        stage: pd.read_json(output / "metrics" / f"{stage}_answers.jsonl", lines=True).set_index("id")
        for stage in ("sft", "dpo")
    }
    for item_id in ("test_04", "test_15"):
        print(answers["sft"].loc[item_id, "question"])
        for stage in ("sft", "dpo"):
            print(f"\\n{stage.upper()}\\n{answers[stage].loc[item_id, 'answer']}")
        print()
""")
    notebook = {"cells": cells, "metadata": {"colab": {"gpuType": "T4", "provenance": []},
                 "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "autoservice_dpo.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
