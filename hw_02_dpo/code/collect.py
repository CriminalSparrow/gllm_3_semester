"""Получить варианты ответов от настоящего адаптера РК1."""

import argparse
from pathlib import Path

from model import environment, generate, load_policy
from rk2 import CONFIG, SFT, digest, preference_prompts, read_json, validate_splits, write_json


def collect(output, backend, base_path=None, batch_size=8, limit=None):
    validate_splits()
    rows = preference_prompts()
    if limit:
        rows = rows[:limit]
    path = output / "logs/preference_generation.json"
    metadata = {"environment": environment(backend), "batch_size": batch_size,
                "seed": CONFIG["seed"], "top_p": CONFIG["top_p"],
                "max_new_tokens": CONFIG["candidate_max_new_tokens"],
                "temperatures": CONFIG["candidate_temperatures"], "prompt_ids": [r["id"] for r in rows],
                "prompt_split_sha256": {split: digest(SFT / f"data/{split}.jsonl") for split in ("train", "validation")},
                "source_adapter_sha256": digest(SFT / "adapter/adapter_model.safetensors")}
    state = read_json(path) if path.exists() else {"settings": metadata, "items": {}}
    if state["settings"] != metadata:
        raise ValueError(f"Настройки изменились. Для нового сбора укажите другую папку, текущая: {output}")
    model = tokenizer = None
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        for index, temperature in enumerate(CONFIG["candidate_temperatures"]):
            mode = f"c{index}"
            present = [mode in state["items"].get(row["id"], {}).get("candidates", {}) for row in batch]
            if all(present):
                continue
            if any(present):
                raise ValueError("Неполный сохранённый батч кандидатов")
            if model is None:
                model, tokenizer = load_policy(backend, base_path)
            seed = CONFIG["seed"] + index * 1000 + begin
            generated = generate(model, tokenizer, batch, temperature=temperature,
                                 max_tokens=CONFIG["candidate_max_new_tokens"], seed=seed)
            for row, candidate in zip(batch, generated, strict=True):
                item = state["items"].setdefault(row["id"], {**row, "candidates": {}})
                item["candidates"][mode] = {**candidate, "temperature": temperature, "seed": seed}
            write_json(path, state)
            print(f"Кандидаты {mode}, температура {temperature}: {begin + len(batch)}/{len(rows)}", flush=True)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["t4", "mps"], default="t4")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    collect(args.output, args.backend, args.base_path, args.batch_size, args.limit)
