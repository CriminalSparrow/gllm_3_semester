"""Минимальный клиент OpenAI-совместимого API. Секреты не логируются."""
import json
import os
import time
import urllib.request
from pathlib import Path

def load_env():
    for p in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

def complete(messages, max_tokens=2500):
    load_env()
    body = dict(model=os.getenv("JUDGE_MODEL", "glm-4.7"), messages=messages,
                temperature=0, max_tokens=max_tokens, thinking={"type": "disabled"})
    req = urllib.request.Request(os.environ["LLM_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + os.environ["LLM_TOKEN"],
        "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.load(response)
            msg = data["choices"][0]
            if msg["finish_reason"] == "length":
                raise ValueError("API output truncated")
            return {"text": msg["message"]["content"], "model": data.get("model", body["model"]), "requested_model": body["model"],
                    "usage": data.get("usage", {}), "id": data.get("id")}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)
