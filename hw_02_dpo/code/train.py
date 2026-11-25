"""Подбор beta и learning rate через DPOTrainer на preference validation."""

import argparse
import hashlib
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import set_seed
from trl import DPOConfig, DPOTrainer

from model import environment, load_policy, release
from rk2 import CONFIG, SFT, SFT_CONFIG, digest, read_json, validate_splits, write_json


def adapter_hash(model, name):
    sha = hashlib.sha256()
    for key, parameter in sorted(model.named_parameters()):
        if f".{name}." in key:
            sha.update(key.replace(f".{name}.", ".").encode())
            sha.update(parameter.detach().float().cpu().numpy().tobytes())
    return sha.hexdigest()


def format_pairs(dataset, tokenizer):
    def render(row):
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        chosen, rejected = (row[key][0]["content"] for key in ("chosen", "rejected"))
        lengths = [len(tokenizer.encode(x, add_special_tokens=False)) for x in (prompt, chosen, rejected)]
        lengths[1] += 1
        lengths[2] += 1
        if lengths[0] > CONFIG["max_prompt_length"] or max(lengths[1:]) > CONFIG["max_completion_length"]:
            raise ValueError(f"Пример {row['id']} пришлось бы обрезать: {lengths}")
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
                "sequence_tokens": lengths[0] + max(lengths[1:])}

    rendered = {}
    max_tokens = {}
    for split in ("train", "validation"):
        data = dataset[split].map(render, remove_columns=dataset[split].column_names)
        max_tokens[split] = max(data["sequence_tokens"])
        rendered[split] = data.remove_columns("sequence_tokens")
    return rendered, max_tokens


def train_trial(trial, dataset, output, backend, base_path, epochs, batch_size):
    set_seed(CONFIG["seed"])
    trial_dir = output / "runs" / trial["id"]
    model, tokenizer = load_policy(backend, base_path, training=True)
    initial_ref = adapter_hash(model, "reference")
    initial_policy = adapter_hash(model, "default")
    if initial_policy != initial_ref:
        raise RuntimeError("Policy и reference должны стартовать с одинаковыми весами SFT")
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not names or any(".default." not in name for name in names):
        raise RuntimeError("В оптимизатор должны поступить только веса адаптера policy")
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    tables, lengths = format_pairs(dataset, tokenizer)
    args = DPOConfig(
        output_dir=str(trial_dir), loss_type="sigmoid", beta=trial["beta"],
        learning_rate=trial["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=0.1,
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        model_adapter_name="default", ref_adapter_name="reference", disable_dropout=True,
        max_prompt_length=CONFIG["max_prompt_length"], max_completion_length=CONFIG["max_completion_length"],
        max_length=CONFIG["max_prompt_length"] + CONFIG["max_completion_length"],
        optim="adamw_torch", max_grad_norm=1.0, fp16=backend == "t4", bf16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_logits_to_keep=True, precompute_ref_log_probs=True,
        seed=CONFIG["seed"], data_seed=CONFIG["seed"], logging_steps=1,
        eval_strategy="epoch", save_strategy="no", report_to="none", remove_unused_columns=False,
        dataloader_pin_memory=backend == "t4", disable_tqdm=True,
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=tables["train"],
                         eval_dataset=tables["validation"], processing_class=tokenizer)
    initial = trainer.evaluate()
    if abs(initial["eval_loss"] - math.log(2)) > 0.03 or abs(initial["eval_rewards/margins"]) > 0.05:
        raise RuntimeError(f"Ожидались исходные loss=ln(2) и reward margin=0: {initial}")
    if backend == "t4":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    trained = trainer.train()
    elapsed = time.monotonic() - start
    final = next(entry for entry in reversed(trainer.state.log_history) if "eval_loss" in entry)
    if not all(math.isfinite(float(final[key])) for key in ("eval_loss", "eval_rewards/margins")):
        raise RuntimeError(f"Неконечные метрики DPO: {final}")
    if adapter_hash(model, "reference") != initial_ref or adapter_hash(model, "default") == initial_policy:
        raise RuntimeError("Reference изменился или policy остался в исходном состоянии")
    model.peft_config["default"].base_model_name_or_path = SFT_CONFIG["model"]
    model.peft_config["default"].revision = SFT_CONFIG["revision"]
    model.save_pretrained(trial_dir / "adapter", selected_adapters=["default"], safe_serialization=True)
    stats = {"config": trial, "before": initial, "after": final, "history": trainer.state.log_history,
             "train": trained.metrics, "training_seconds": elapsed, "max_sequence_tokens": lengths,
             "trainable_params": trainable_params, "reference_unchanged": True, "policy_changed": True,
             "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3 if backend == "t4" else None}
    del trainer, model
    release(backend)
    return stats


def train(preference_dir, output, backend, base_path=None, epochs=None, batch_size=None):
    validate_splits()
    epochs = epochs or CONFIG["epochs"]
    batch_size = batch_size or CONFIG["batch_size"]
    paths = {split: preference_dir / f"preference_{split}.jsonl" for split in ("train", "validation")}
    dataset = load_dataset("json", data_files={key: str(path) for key, path in paths.items()})
    if not len(dataset["train"]) or not len(dataset["validation"]):
        raise ValueError("Для DPO нужны размеченные train и validation пары")
    if set(dataset["train"]["id"]) & set(dataset["validation"]["id"]):
        raise ValueError("ID train и validation пересекаются")
    log_path = output / "logs/training.json"
    settings = {"environment": environment(backend), "epochs": epochs, "batch_size": batch_size,
                "gradient_accumulation": CONFIG["gradient_accumulation"], "seed": CONFIG["seed"],
                "train_size": len(dataset["train"]), "validation_size": len(dataset["validation"]),
                "data_sha256": {key: digest(path) for key, path in paths.items()},
                "sft_adapter_sha256": digest(SFT / "adapter/adapter_model.safetensors"),
                "trials": CONFIG["trials"]}
    state = read_json(log_path) if log_path.exists() else {"settings": settings, "trials": {}}
    if state["settings"] != settings:
        raise ValueError("Для изменённых параметров используйте новую папку результатов")
    for trial in CONFIG["trials"]:
        key = trial["id"]
        if key in state["trials"] and (output / "runs" / key / "adapter/adapter_model.safetensors").exists():
            continue
        print(f"DPO trial {trial}", flush=True)
        state["trials"][key] = train_trial(trial, dataset, output, backend, base_path, epochs, batch_size)
        write_json(log_path, state)
    # Reward accuracy compares validation rankings without preferring the scale of any beta.
    best = min(state["trials"], key=lambda key: (
        -state["trials"][key]["after"]["eval_rewards/accuracies"],
        state["trials"][key]["after"]["eval_loss"], key))
    adapter = output / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        source = output / "runs" / best / "adapter" / name
        temp = (adapter / name).with_suffix(".tmp")
        temp.write_bytes(source.read_bytes())
        temp.replace(adapter / name)
    state["selected"] = {"id": best, "rule": "max validation reward accuracy; tie: min validation DPO loss",
                         "adapter_sha256": digest(adapter / "adapter_model.safetensors")}
    write_json(log_path, state)
    print(f"DPO_COMPLETE {best}", flush=True)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["t4", "mps"], default="t4")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    train(args.preference_dir, args.output, args.backend, args.base_path, args.epochs, args.batch_size)
