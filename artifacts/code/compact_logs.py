"""Однократно объединить старые логи; новые скрипты сразу пишут общие JSON."""
import hashlib
import json
from common import ROOT

def main():
    folder = ROOT / "logs"
    names = ["data_preparation", "judge", "training"]
    bundles = {name: json.loads((folder / f"{name}.json").read_text())
               if (folder / f"{name}.json").exists() else {} for name in names}
    # В старом training.json находился только словарь метрик.
    if "train_runtime" in bundles["training"]:
        bundles["training"] = {"metrics": bundles["training"]}
    legacy = [p for p in folder.iterdir() if p.is_file() and p.name not in
              ["data_preparation.json", "judge.json", "training.json"]]
    for path in sorted(legacy):
        name = path.name
        text = path.read_text()
        value = json.loads(text) if path.suffix == ".json" else (
            [json.loads(s) for s in text.splitlines() if s.strip()] if path.suffix == ".jsonl" else text)
        if name.startswith("generation_"):
            target, key, entry = "data_preparation", "generations", path.stem.removeprefix("generation_")
        elif name.startswith("rejected_generation_"):
            target, key, entry = "data_preparation", "rejected_generations", path.stem.removeprefix("rejected_generation_")
        elif name == "data_editorial_edits.jsonl":
            target, key, entry = "data_preparation", "editorial_edits", None
        elif name.startswith("prepare_data"):
            target, key, entry = "data_preparation", "console", name
        elif name.startswith("judge_raw_"):
            target, key, entry = "judge", "raw_responses", path.stem.removeprefix("judge_raw_")
        elif name.startswith("judge_test_"):
            target, key, entry = "judge", "evaluations", path.stem.removeprefix("judge_")
        elif name.startswith("judge"):
            target, key, entry = "judge", "console", name
        else:
            target = "training"
            key, entry = {"environment.json": ("environment", None),
                "trainer_history.jsonl": ("history", None), "pip_freeze.txt": ("pip_freeze", None),
                "experiment.log": ("console", None), "verification.txt": ("verification", None)}.get(name, ("legacy_files", name))
        if entry is None:
            bundles[target][key] = value
        else:
            bundles[target].setdefault(key, {})[entry] = value
        bundles[target].setdefault("source_files", {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, data in bundles.items():
        destination = folder / f"{name}.json"
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        assert json.loads(destination.read_text()) == data
    for path in legacy:
        path.unlink()
    print(f"Логи объединены: {len(legacy) + 1} старых файлов → 3 JSON")

if __name__ == "__main__":
    main()
