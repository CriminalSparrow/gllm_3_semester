"""Графики динамики обучения и итогового сравнения моделей."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rk2 import ROOT, read_json


def plot_training(output):
    log = read_json(output / "logs/training.json")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
    axes = axes.flatten()
    fields = ["loss", "eval_loss", "rewards/margins", "eval_rewards/accuracies"]
    labels = ["Train DPO loss", "Validation DPO loss", "Train reward margin", "Validation reward accuracy"]
    for key, trial in log["trials"].items():
        config = trial["config"]
        label = f"β={config['beta']:g}, LR={config['learning_rate']:g}"
        width = 2.5 if key == log["selected"]["id"] else 1.3
        for axis, field in zip(axes, fields, strict=True):
            initial = [(0, trial["before"][field])] if field in trial["before"] else []
            data = initial + [(row["step"], row[field]) for row in trial["history"] if field in row]
            x, y = zip(*data)
            axis.plot(x, y, marker="o", markersize=3, linewidth=width, label=label)
    for axis, label in zip(axes, labels, strict=True):
        axis.set(title=label, xlabel="Шаг оптимизатора")
        axis.grid(alpha=0.23)
    axes[-1].set_ylim(-0.04, 1.04)
    axes[0].legend(fontsize=8)
    fig.suptitle("DPO на preference-парах автосервиса")
    directory = output / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / "training.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    selected = log["trials"][log["selected"]["id"]]
    for stage, prefix, style in [("Train", "", "-"), ("Validation", "eval_", "--")]:
        for kind, color in [("chosen", "#2664ad"), ("rejected", "#bd4e32")]:
            key = f"{prefix}rewards/{kind}"
            initial = [(0, selected["before"][key])] if key in selected["before"] else []
            entries = initial + [(row["step"], row[key]) for row in selected["history"] if key in row]
            x, y = zip(*entries)
            axis.plot(x, y, style, color=color, marker="o", markersize=3, label=f"{stage} {kind}")
    axis.set(title="Награды выбранной конфигурации", xlabel="Шаг оптимизатора", ylabel="β · Δ log P")
    axis.axhline(0, color="0.6", linewidth=0.8)
    axis.grid(alpha=0.23)
    axis.legend()
    fig.savefig(directory / "rewards.png", dpi=160)
    plt.close(fig)

    with (output / "metrics/trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["id", "beta", "learning_rate", "validation_loss", "validation_margin",
                         "validation_accuracy", "seconds", "peak_cuda_gib", "selected"])
        for name, trial in log["trials"].items():
            cfg, end = trial["config"], trial["after"]
            writer.writerow([name, cfg["beta"], cfg["learning_rate"], end["eval_loss"],
                             end["eval_rewards/margins"], end["eval_rewards/accuracies"],
                             trial["training_seconds"], trial["peak_cuda_gib"], name == log["selected"]["id"]])


def plot_evaluation(output):
    summary = read_json(output / "metrics/summary.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    groups = [("overall", "Все"), ("rk1", "РК1"), ("fresh", "Новые")]
    runs = [(label, summary[key]) for key, label in groups if summary[key]["n"]]
    prior = [0] * len(runs)
    for model, color, label in [("sft", "#607891", "SFT"), ("tie", "#ccbca5", "Ничья"), ("dpo", "#276d59", "DPO")]:
        widths = [stats["wins"][model] / stats["n"] * 100 for _, stats in runs]
        left.barh([name for name, _ in runs], widths, left=prior, color=color, label=label)
        prior = [a + b for a, b in zip(prior, widths, strict=True)]
    left.set(xlim=(0, 100), xlabel="Доля вопросов, %", title="Победы по сумме трёх шкал")
    left.invert_yaxis()
    left.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    axes = [("safety", "Безопасность"), ("correctness", "Корректность"), ("usefulness", "Полезность")]
    for i, (model, color) in enumerate([("sft", "#607891"), ("dpo", "#276d59")]):
        right.bar([n + (i - 0.5) * 0.34 for n in range(3)],
                  [summary["overall"]["axes"][metric][model] for metric, _ in axes],
                  width=0.34, color=color, label=model.upper())
    right.set_xticks(range(3), [label for _, label in axes], rotation=15)
    right.set(ylim=(0, 4), ylabel="Средний балл", title="Слепая оценка ответов")
    right.legend(fontsize=8)
    right.grid(axis="y", alpha=0.22)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "plots/comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=["training", "evaluation", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ("all", "training"):
        plot_training(args.output)
    if args.stage in ("all", "evaluation"):
        plot_evaluation(args.output)
