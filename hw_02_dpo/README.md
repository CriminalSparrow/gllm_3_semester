# РК2. DPO для консультанта автосервиса

[Отчёт](REPORT.md) · [Ноутбук](autoservice_dpo.ipynb) · [РК1](../hw_01_sft/README.md)

DPO продолжает обучение LoRA из РК1 на парах ответов той же SFT-модели. База и замороженный reference используют Qwen2.5-1.5B-Instruct той же ревизии. На тесте SFT и DPO получают одинаковые вопросы, системный промпт, greedy-декодирование и лимит генерации.

## Google Colab

Откройте `autoservice_dpo.ipynb` с GitHub или загрузите файл в Colab. Параметр `MODE` во второй кодовой ячейке определяет запуск.

| Режим | Действие |
|---|---|
| `replay` | Загрузить материалы репозитория, показать сохранённые пары, журнал, метрики и примеры |
| `collect` | Выбрать T4 и получить новых SFT-кандидатов и задания для слепой разметки |
| `train` | Выбрать T4, обучить три конфигурации на сохранённых парах, получить тестовые ответы и слепой пакет |

Выбор ускорителя находится в **Среда выполнения → Сменить среду выполнения → T4**. Запускайте ячейки сверху вниз. Colab скачивает базовые веса с Hugging Face. Адаптер РК1 уже хранится в `hw_01_sft/adapter/`. Результаты повторного запуска сохраняются в `hw_02_dpo/review_run/`.

Зафиксированный эксперимент запущен на Apple M3 Pro с FP16-базой, поскольку Colab не выделил T4. Для такого же запуска установите PyTorch с MPS и добавьте `--backend mps` к `collect.py`, `train.py` и `predict.py`. Если база уже скачана, передайте её папку через `--base-path`. В исходном прогоне батч генерации равен 16 для сбора и 12 для теста. Код для T4 загружает ту же ревизию в NF4. Результаты на двух профилях могут различаться.

## Скрипты

Команды из корня репозитория. Для сокращения команд используется временная папка `hw_02_dpo/review_run/cli`.

```bash
pip install -r hw_02_dpo/requirements.txt
RK2=hw_02_dpo
OUT=hw_02_dpo/review_run/cli

# Сгенерировать новые ответы и обезличить позиции для судьи
python "$RK2/code/collect.py" --output "$OUT" --batch-size 16
python "$RK2/code/rank.py" prepare --generation "$OUT/logs/preference_generation.json" \
  --package "$OUT/blind_preferences.jsonl" --mapping "$OUT/preferences_mapping.json"

# Собрать пары после выставления оценок судьёй
python "$RK2/code/rank.py" assemble --generation "$OUT/logs/preference_generation.json" \
  --package "$OUT/blind_preferences.jsonl" --mapping "$OUT/preferences_mapping.json" \
  --ratings "$OUT/preference_ratings.jsonl" --output "$OUT"

# Для повторения зафиксированного эксперимента используйте $RK2/data вместо $OUT/data
python "$RK2/code/train.py" --preference-dir "$OUT/data" --output "$OUT"
python "$RK2/code/predict.py" --output "$OUT"
python "$RK2/code/evaluate.py" prepare --output "$OUT"

# Посчитать метрики после оценки новых тестовых ответов
python "$RK2/code/evaluate.py" aggregate --output "$OUT" --scores "$OUT/test_ratings.jsonl"
python "$RK2/code/plot.py" --output "$OUT"
```

Передайте судье только слепой пакет и рубрику `RUBRIC` из `rank.py` или `evaluate.py`. Разметка preference представляет JSONL вида `{"id":"...","scores":{"A":{"safety":3,"correctness":3,"usefulness":3,"note":"..."},"B":{...}}}` с оценками всех показанных вариантов. Разметка теста использует `{"id":"...","A":{...},"B":{...}}`. Баллы целые от 0 до 4. Для пересчёта сохранённых результатов используйте `ratings` из `logs/preference.json` и `logs/evaluation.json`. Для первого файла потребуется преобразовать отображение `id → scores` в строки `{"id": id, "scores": scores}`, для второго достаточно записать каждый элемент массива отдельной строкой. Новый текст ответа требует новой оценки. API-ключи для этого эксперимента не использовались.

Загрузить данные из корня репозитория можно через `datasets.load_dataset`.

```python
from datasets import load_dataset

pairs = load_dataset("json", data_files={
    "train": "hw_02_dpo/data/preference_train.jsonl",
    "validation": "hw_02_dpo/data/preference_validation.jsonl",
})
print(pairs["train"][0]["prompt"], pairs["train"][0]["chosen"])
```

## Артефакты

| Путь | Содержимое |
|---|---|
| `data/preference_*.jsonl` | Диалоговые `prompt`, `chosen`, `rejected` для обучения и валидации |
| `data/test_fresh.jsonl` | 24 новых тестовых вопроса; 20 старых берутся из `hw_01_sft/data/test.jsonl` |
| `adapter/` | Итоговый LoRA-адаптер в `safetensors` и конфигурация PEFT |
| `logs/preference_generation.json`, `logs/preference.json` | Кандидаты, параметры, оценки судьи, отбор и исключения |
| `logs/training.json` | Окружение, гиперпараметры, loss, rewards и правило выбора |
| `logs/prediction.json`, `logs/evaluation.json` | Происхождение тестовых ответов и слепая разметка |
| `metrics/`, `plots/` | Ответы моделей, таблицы оценок и сравнения, графики |

Подключение сохранённого DPO использует `PeftModel.from_pretrained(base_model, 'hw_02_dpo/adapter')`. Полный пример загрузки базы и профили T4 NF4 и Apple Metal FP16 находятся в `code/model.py`. DPO-адаптер содержит параметры после продолжения SFT и подключается напрямую к базе.
