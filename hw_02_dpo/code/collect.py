"""Получить варианты ответов от настоящего адаптера РК1."""

import argparse
from pathlib import Path

from model import GUIDANCE, environment, generate, load_policy
from rk2 import CONFIG, SFT, digest, guided_prompts, legacy_preference_prompts, read_json, validate_splits, write_json


def collect(output, backend, base_path=None, batch_size=8, limit=None):
    validate_splits()
    rows = legacy_preference_prompts()
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


def collect_guided(output, backend, base_path=None, batch_size=24, paths=None, archive=None):
    rows = guided_prompts(paths)
    path = output / "logs/preference_generation.json"
    state = read_json(path if path.exists() else archive) if path.exists() or archive else {"items": {}}
    parameters = {"environment": environment(backend), "batch_size": batch_size, "seed": CONFIG["seed"],
                  "max_new_tokens": CONFIG["guided_max_new_tokens"], "decoding": "greedy",
                  "guidance": GUIDANCE, "source_adapter_sha256": digest(SFT / "adapter/adapter_model.safetensors")}
    if state.setdefault("guided_settings", parameters) != parameters:
        raise ValueError("Параметры расширенного сбора изменились")
    pending = []
    for row in rows:
        previous = state["items"].get(row["id"])
        if previous is None:
            pending.append(row)
        elif any(previous[k] != value for k, value in row.items()) or set(previous["candidates"]) != {"c0", "c1"}:
            raise ValueError(f"Исходный вопрос изменился: {row['id']}")
    if not pending:
        return state
    model, tokenizer = load_policy(backend, base_path)
    for begin in range(0, len(pending), batch_size):
        batch = pending[begin:begin + batch_size]
        generations = {}
        for key, guided in (("c0", False), ("c1", True)):
            inputs = [{**row, "guided": guided} for row in batch]
            generations[key] = generate(model, tokenizer, inputs, max_tokens=parameters["max_new_tokens"],
                                        seed=CONFIG["seed"])
        for i, row in enumerate(batch):
            state["items"][row["id"]] = {**row, "protocol": "guided", "candidates": {
                key: {**values[i], "guided": key == "c1"} for key, values in generations.items()}}
        write_json(path, state)
        print(f"Два ответа SFT: {begin + len(batch)}/{len(pending)}", flush=True)
    return state


def merge_generation(paths, output):
    merged = {"items": {}, "guided_runs": []}
    for path in paths:
        shard = read_json(path)
        if overlap := merged["items"].keys() & shard["items"].keys():
            raise ValueError(f"Повторные ID между логами: {sorted(overlap)}")
        if "settings" in shard:
            merged["settings"] = shard["settings"]
        ids = [key for key, row in shard["items"].items() if row.get("protocol") == "guided"]
        if ids:
            merged["guided_runs"].append({"settings": shard["guided_settings"], "prompt_ids": ids})
        merged["items"].update(shard["items"])
    write_json(output / "logs/preference_generation.json", merged)
    print(f"Объединено вопросов: {len(merged['items'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["t4", "mps"], default="t4")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--guided-only", action="store_true")
    parser.add_argument("--guided-batch-size", type=int, default=24)
    parser.add_argument("--pool", type=Path, action="append")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+")
    args = parser.parse_args()
    validate_splits()
    if args.merge:
        merge_generation(args.merge, args.output)
    else:
        if not args.guided_only:
            collect(args.output, args.backend, args.base_path, args.batch_size, args.limit)
        if args.limit is None:
            collect_guided(args.output, args.backend, args.base_path, args.guided_batch_size, args.pool, args.archive)
