"""Ответ сохранённой модели без повторного обучения: python infer.py --question '...'"""
import argparse
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import ROOT, SYSTEM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="После ночи стартер щёлкает, фары тускнеют. Что делать?")
    args = parser.parse_args()
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), "Выберите GPU T4"
    cfg = json.loads((ROOT / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "adapter")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"],
        quantization_config=quant, device_map={"": 0}, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ROOT / "adapter").eval()
    ids = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.question}], add_generation_prompt=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=384, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
