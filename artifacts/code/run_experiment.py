"""Один небольшой QLoRA-эксперимент, только на Colab T4."""
import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
import torch
import transformers
import peft
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
from common import ROOT, MODEL, SEED, SYSTEM, read_jsonl, write_jsonl, check_answer, check_examples, save_log

MAX_LENGTH = 768

def main():
    assert torch.cuda.is_available(), "Нужен GPU T4: Среда выполнения → Сменить среду выполнения"
    gpu = torch.cuda.get_device_name(0)
    assert "T4" in gpu, f"Для этого задания выберите T4, получен {gpu}"
    set_seed(SEED)
    start = time.time()
    env = {"gpu": gpu, "python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "peft": peft.__version__, "seed": SEED}
    save_log("training", "environment", env)
    save_log("training", "pip_freeze", subprocess.check_output(["python", "-m", "pip", "freeze"], text=True))
    write_jsonl(ROOT / "metrics/check_unit_cases.jsonl", check_examples())
    config_path = ROOT / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    revision = config.get("revision", "main")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    def load_base():
        return AutoModelForCausalLM.from_pretrained(MODEL, revision=revision,
            quantization_config=quant, device_map={"": 0}, torch_dtype=torch.float16,
            attn_implementation="sdpa")
    model = load_base()
    revision = model.config._commit_hash or revision
    config = {"model": MODEL, "revision": revision, "seed": SEED, "max_length": MAX_LENGTH,
              "epochs": 2, "batch_size": 2, "gradient_accumulation": 4, "lr": 2e-4,
              "lora_r": 8, "lora_alpha": 16, "max_new_tokens": 384,
              "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    config_path.write_text(json.dumps(config, indent=2))
    test = read_jsonl(ROOT / "data/test.jsonl")

    def encode(row):
        messages = row["messages"]
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        assert len(full) <= MAX_LENGTH, f"Слишком длинный пример: {row['id']} ({len(full)} токенов)"
        assert full[:len(prefix)] == prefix
        return {"input_ids": full, "attention_mask": [1] * len(full),
                "labels": [-100] * len(prefix) + full[len(prefix):]}

    train = Dataset.from_list([encode(r) for r in read_jsonl(ROOT / "data/train.jsonl")])
    validation = Dataset.from_list([encode(r) for r in read_jsonl(ROOT / "data/validation.jsonl")])
    print("Data sanity:", len(train), len(validation), "max tokens:", max(map(len, train["input_ids"])), flush=True)

    def collate(rows):
        width = max(len(r["input_ids"]) for r in rows)
        return {key: torch.tensor([r[key] + [pad] * (width - len(r[key])) for r in rows])
                for key, pad in [("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)]}

    def predict(m, stage):
        m.eval(); m.config.use_cache = True
        tokenizer.padding_side = "left"
        results = []
        for begin in range(0, len(test), 4):
            batch = test[begin:begin + 4]
            prompts = [tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True) for r in batch]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = m.generate(**inputs, max_new_tokens=384, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            for row, output in zip(batch, ids[:, inputs.input_ids.shape[1]:]):
                answer = tokenizer.decode(output, skip_special_tokens=True).strip()
                eos = tokenizer.eos_token_id in output.tolist()
                results.append({**row, "stage": stage, "answer": answer,
                                "ended_with_eos": eos, **check_answer(answer)})
            print(stage, len(results), "/", len(test), flush=True)
        write_jsonl(ROOT / f"metrics/{stage}_answers.jsonl", results)
        tokenizer.padding_side = "right"
        return results

    # До обучения: те же вопросы, промпт, квантизация и greedy decoding.
    before = predict(model, "before")
    print("Sanity baseline example:", before[0]["answer"], flush=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=config["targets"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(ROOT / "checkpoints"), num_train_epochs=2,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, warmup_ratio=0.1, lr_scheduler_type="cosine", optim="adamw_torch",
        fp16=True, gradient_checkpointing=True, logging_steps=2, eval_strategy="epoch",
        save_strategy="no", report_to="none", seed=SEED, data_seed=SEED,
        remove_unused_columns=False, max_grad_norm=1.0)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=validation, data_collator=collate)
    # Небольшая проверка masked loss до полного прогона.
    batch = {k: v.to("cuda") for k, v in collate([train[0]]).items()}
    model.eval()
    with torch.no_grad():
        sanity_loss = model(**batch).loss.item()
    assert 0 < sanity_loss < 20, sanity_loss
    print("Sanity masked loss:", sanity_loss, flush=True)
    baseline_loss = trainer.evaluate()["eval_loss"]
    training_start = time.time()
    result = trainer.train()
    training_seconds = time.time() - training_start
    final_loss = trainer.evaluate()["eval_loss"]
    model.save_pretrained(ROOT / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(ROOT / "adapter")
    save_log("training", "history", trainer.state.log_history)
    save_log("training", "metrics", {**result.metrics, "sanity_loss": sanity_loss,
        "before_validation_loss": baseline_loss, "after_validation_loss": final_loss,
        "training_seconds": training_seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated()/1024**3})
    del trainer, model, batch
    gc.collect(); torch.cuda.empty_cache()
    # Оценка загруженного с диска адаптера проверяет пригодность сдаваемых весов.
    model = PeftModel.from_pretrained(load_base(), ROOT / "adapter")
    after = predict(model, "after")
    summary = {stage: {"n": len(rows), "format_pass_rate": sum(r["pass"] for r in rows)/len(rows),
        "mean_words": sum(r["words"] for r in rows)/len(rows),
        "truncated": sum(not r["ended_with_eos"] for r in rows)} for stage, rows in [("before", before), ("after", after)]}
    summary["elapsed_seconds"] = time.time() - start
    (ROOT / "metrics/automatic_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("EXPERIMENT_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
