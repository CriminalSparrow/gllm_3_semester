"""Общие настройки и одна автоматическая проверка ответа."""
import json
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
_LOG_LOCK = threading.RLock()
HEADINGS = ["Возможные причины:", "Что делать:", "Срочность:", "Уточнить:"]
SYSTEM = (
    "Ты консультант автосервиса. Отвечай по-русски, кратко (до 150 слов). "
    "Используй ровно четыре раздела: Возможные причины:, Что делать:, Срочность:, Уточнить:. "
    "Указывай возможные причины, а не достоверный диагноз. Не выдумывай цены и регламенты. "
    "При опасных симптомах укажи, можно ли продолжать движение. "
    "Не предлагай опасный ремонт без специалиста. Задай 1–2 уточняющих вопроса."
)

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

def read_log(name):
    path = ROOT / "logs" / (name + ".json")
    with _LOG_LOCK:
        return json.loads(path.read_text()) if path.exists() else {}

def save_log(name, key, value, entry=None):
    """Обновить общий JSON; потоки генератора/судьи не затирают друг друга."""
    with _LOG_LOCK:
        data = read_log(name)
        if entry is None:
            data[key] = value
        else:
            data.setdefault(key, {})[entry] = value
        path = ROOT / "logs" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

def display_answer(text):
    """Отдельный абзац для каждого раздела; исходный текст ответа не меняется."""
    for heading in HEADINGS:
        text = re.sub(r"\s*" + re.escape(heading), "\n\n" + heading, text)
    return text.strip()

def model_check_examples():
    """Два настоящих отказа базы и два прошедших ответа адаптера."""
    rows = []
    for stage in ["before", "after"]:
        answers = read_jsonl(ROOT / f"metrics/{stage}_answers.jsonl")
        for answer in answers:
            if answer["id"] in ["test_01", "test_11"]:
                rows.append({"id": answer["id"], "stage": stage,
                    "question": answer["question"], "answer": answer["answer"],
                    **check_answer(answer["answer"])})
    assert [r["pass"] for r in rows] == [False, False, True, True]
    return rows

def check_answer(text):
    """Контракт консультации: 4 непустых раздела по порядку, <=150 слов."""
    words = len(text.split())
    reasons = []
    if not 20 <= words <= 150:
        reasons.append("length")
    pattern = r"^\s*" + r"\s*".join(re.escape(h) + r"\s*(.+?)" for h in HEADINGS) + r"\s*$"
    match = re.match(pattern, text, re.S)
    if not match or any(not s.strip() for s in match.groups()) or any(text.count(h) != 1 for h in HEADINGS):
        reasons.append("sections")
    return {"pass": not reasons, "words": words, "reasons": reasons}

def check_examples():
    good = (
        "Возможные причины: Разряжен аккумулятор или плохой контакт клемм.\n"
        "Что делать: Проверьте клеммы при выключенном зажигании, обратитесь в сервис.\n"
        "Срочность: До диагностики не рассчитывайте на повторный запуск.\n"
        "Уточнить: Стартер щёлкает? Тускнеют ли фары при запуске?"
    )
    cases = [
        ("correct", good, True),
        ("missing_section", good.split("Уточнить:")[0], False),
        ("too_long", good + " Проверьте." * 150, False),
        ("empty_section", good.replace("Срочность: До диагностики не рассчитывайте на повторный запуск.", "Срочность:"), False),
        ("duplicate_section", good + "\nУточнить: Какой пробег?", False),
        ("wrong_order", good.replace("Что делать:", "TEMP:").replace("Срочность:", "Что делать:").replace("TEMP:", "Срочность:"), False),
    ]
    rows = [{"case": name, "answer": a, "expected": e, **check_answer(a)} for name, a, e in cases]
    assert all(r["pass"] == r["expected"] for r in rows)
    return rows

if __name__ == "__main__":
    write_jsonl(ROOT / "metrics/check_unit_cases.jsonl", check_examples())
    if (ROOT / "metrics/after_answers.jsonl").exists():
        write_jsonl(ROOT / "metrics/check_examples.jsonl", model_check_examples())
    print("6 граничных случаев и 4 реальных примера: OK")
