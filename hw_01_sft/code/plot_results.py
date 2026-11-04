"""Графики из фактических логов, без повторного обучения."""
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, read_jsonl, read_log

def main():
    auto = json.loads((ROOT / "metrics/automatic_summary.json").read_text())
    judge = json.loads((ROOT / "metrics/judge_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    stages = ["before", "after"]
    axes[0].bar(stages, [100*auto[s]["format_pass_rate"] for s in stages], color=["#9ba8b4", "#338c77"])
    axes[0].set(ylim=(0, 105), title="Format pass rate", ylabel="% of 20 questions")
    keys = ["safety", "correctness", "usefulness"]
    for j, stage in enumerate(stages):
        axes[1].bar([i + (j-.5)*.36 for i in range(3)], [judge[stage][k] for k in keys], width=.36, label=stage)
    axes[1].set(xticks=range(3), xticklabels=keys, ylim=(0, 4.4), title=judge['judge_model'] + " judge", ylabel="Mean score (0–4)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "plots/comparison.png", dpi=170); plt.close(fig)
    history = read_log("training")["history"]
    train = [r for r in history if "loss" in r]
    val = [r for r in history if "eval_loss" in r]
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot([r["step"] for r in train], [r["loss"] for r in train], "o-", label="train")
    ax.plot([r["step"] for r in val], [r["eval_loss"] for r in val], "s--", label="validation")
    ax.set(xlabel="Optimizer step", ylabel="Assistant-token cross-entropy", title="QLoRA training")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "plots/loss.png", dpi=170); plt.close(fig)
    with (ROOT / "metrics/summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "n", "format_pass_rate", "mean_words", "truncated", *keys, "total"])
        w.writeheader()
        for stage in stages:
            w.writerow({"stage": stage, **auto[stage], **judge[stage]})

if __name__ == "__main__":
    main()
