"""База и адаптеры РК1/РК2, единый промпт и декодирование."""

import gc
import os
import platform
from importlib.metadata import version

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from rk2 import CONFIG, SFT, SFT_CONFIG, check_answer, messages


def environment(backend):
    gpu = torch.cuda.get_device_name(0) if backend == "t4" else platform.machine()
    packages = ["torch", "transformers", "peft", "accelerate", "datasets", "trl"]
    if backend == "t4":
        packages.append("bitsandbytes")
    return {"backend": backend, "device": gpu, "system": platform.platform(),
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "packages": {name: version(name) for name in packages},
            "base": SFT_CONFIG["model"], "revision": SFT_CONFIG["revision"]}


def load_policy(backend, base_path=None, adapter=None, training=False):
    source = str(base_path or SFT_CONFIG["model"])
    kwargs = {"revision": SFT_CONFIG["revision"], "torch_dtype": torch.float16,
              "attn_implementation": "sdpa"}
    if backend == "t4":
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Выберите Tesla T4 в настройках среды Google Colab")
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16))
    elif backend == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Для локального FP16 прогона нужен Apple Metal (MPS)")
        kwargs["device_map"] = {"": "mps"}
    else:
        raise ValueError(backend)
    base = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if training and backend == "t4":
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    if training:
        base.config.use_cache = False
    adapter = adapter or SFT / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(SFT / "adapter")
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base, adapter, is_trainable=training)
    if training:
        model.load_adapter(SFT / "adapter", adapter_name="reference", is_trainable=False)
        # PEFT can round an added adapter through the base dtype before upcasting it.
        state = get_peft_model_state_dict(model, adapter_name="default")
        set_peft_model_state_dict(model, state, adapter_name="reference")
        model.set_adapter("default")
    return model, tokenizer


def generate(model, tokenizer, rows, *, temperature=0.0, max_tokens=384, seed=42):
    set_seed(seed)
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    prompts = [tokenizer.apply_chat_template(messages(row["question"]), tokenize=False,
                                             add_generation_prompt=True) for row in rows]
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    generation = {"max_new_tokens": max_tokens, "pad_token_id": tokenizer.pad_token_id,
                  "eos_token_id": tokenizer.eos_token_id, "do_sample": temperature > 0}
    if temperature > 0:
        generation.update(temperature=temperature, top_p=CONFIG["top_p"], top_k=0)
    else:
        generation.update(temperature=1.0, top_p=1.0, top_k=50)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation)[:, inputs.input_ids.shape[1]:].cpu()
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []
    for row, output, text in zip(rows, outputs.tolist(), texts, strict=True):
        eos = tokenizer.eos_token_id in output
        token_count = output.index(tokenizer.eos_token_id) + 1 if eos else len(output)
        text = text.strip()
        results.append({"id": row["id"], "answer": text, "ended_with_eos": eos,
                        "tokens": token_count, **check_answer(text)})
    return results


def release(backend):
    gc.collect()
    if backend == "t4":
        torch.cuda.empty_cache()
    else:
        torch.mps.empty_cache()
