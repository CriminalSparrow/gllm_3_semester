"""Получить ответы SFT и выбранного DPO на одинаковом тесте."""

import argparse
from pathlib import Path

from model import environment, generate, load_policy, release
from rk2 import CONFIG, ROOT, SFT, SFT_CONFIG, digest, read_json, test_prompts, validate_splits, write_json, write_jsonl


def predict(output, backend, base_path=None, stage=None, batch_size=4):
    validate_splits()
    rows = test_prompts()
    settings = {"environment": environment(backend), "batch_size": batch_size,
                "seed": CONFIG["seed"], "decoding": "greedy", "max_new_tokens": SFT_CONFIG["max_new_tokens"],
                "prompt_ids": [r["id"] for r in rows],
                "test_sha256": {"rk1": digest(SFT / "data/test.jsonl"),
                                "fresh": digest(ROOT / "data/test_fresh.jsonl")}}
    path = output / "logs/prediction.json"
    state = read_json(path) if path.exists() else {"settings": settings, "answers": {}}
    if state["settings"] != settings:
        raise ValueError("Для других настроек используйте новую папку результатов")
    for name in ([stage] if stage else ["sft", "dpo"]):
        adapter = SFT / "adapter" if name == "sft" else output / "adapter"
        adapter_sha = digest(adapter / "adapter_model.safetensors")
        record = state["answers"].setdefault(name, {"adapter_sha256": adapter_sha, "items": {}})
        if record["adapter_sha256"] != adapter_sha:
            raise ValueError(f"Адаптер {name} поменялся. Используйте новую папку результатов")
        model = tokenizer = None
        for begin in range(0, len(rows), batch_size):
            batch = rows[begin:begin + batch_size]
            if all(row["id"] in record["items"] for row in batch):
                continue
            if model is None:
                model, tokenizer = load_policy(backend, base_path, adapter=adapter)
            answers = generate(model, tokenizer, batch, max_tokens=SFT_CONFIG["max_new_tokens"], seed=CONFIG["seed"])
            for row, answer in zip(batch, answers, strict=True):
                record["items"][row["id"]] = {**row, **answer}
            write_json(path, state)
            print(f"{name.upper()} test: {begin + len(batch)}/{len(rows)}", flush=True)
        del model, tokenizer
        release(backend)
        answers_path = output / f"metrics/{name}_answers.jsonl"
        write_jsonl(answers_path, [record["items"][row["id"]] for row in rows])
        record["answers_sha256"] = digest(answers_path)
        write_json(path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=["t4", "mps"], default="t4")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--stage", choices=["sft", "dpo"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    predict(args.output, args.backend, args.base_path, args.stage, args.batch_size)
