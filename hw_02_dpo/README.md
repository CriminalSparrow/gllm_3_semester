# РК2. DPO для консультанта автосервиса

[Отчёт](REPORT.md) · [Ноутбук](autoservice_dpo.ipynb) · [РК1](../hw_01_sft/README.md)

Датасет содержит 300 preference-пар для 300 разных вопросов автосервиса, из них 270 train и 30 validation. Финальное DPO-обучение и сравнение моделей выполнены в Google Colab на Tesla T4. Для обеих сторон сравнения используются Qwen2.5-1.5B-Instruct в NF4, один системный промпт, greedy-декодирование и лимит 384 новых токена.

## Google Colab

Откройте `autoservice_dpo.ipynb` с GitHub или загрузите файл в Colab. Режим запуска задаёт `MODE` в разделе «Настройки».

| Режим | Действие |
|---|---|
| `replay` | Показать сохранённые пары, графики обучения, результаты теста и примеры |
| `collect` | Выбрать T4 и получить новых SFT-кандидатов и задания для слепой разметки |
| `train` | Выбрать T4, обучить DPO на сохранённых парах, получить тестовые ответы и слепой пакет |

Выбор ускорителя находится в **Среда выполнения → Сменить среду выполнения → T4**. Запускайте ячейки сверху вниз. Colab скачивает базовые веса с Hugging Face. Адаптер РК1 уже хранится в `hw_01_sft/adapter/`. Результаты повторного запуска сохраняются в `hw_02_dpo/review_run/`.

Для локального FP16-прогона установите PyTorch с MPS и добавьте `--backend mps` к `collect.py`, `train.py` и `predict.py`. Папку скачанной базы можно передать через `--base-path`. Профили T4 и MPS используют одинаковую ревизию с разной точностью весов. Фактическая среда каждой части сбора записана в `logs/preference_generation.json`.

## Скрипты

Команды из корня репозитория. Для сокращения команд используется временная папка `hw_02_dpo/review_run/cli`.

```bash
pip install -r hw_02_dpo/requirements.txt
RK2=hw_02_dpo
OUT=hw_02_dpo/review_run/cli

# Сгенерировать новые ответы и обезличить позиции для судьи
python "$RK2/code/collect.py" --output "$OUT" --batch-size 16 --guided-batch-size 32
python "$RK2/code/rank.py" prepare --generation "$OUT/logs/preference_generation.json" \
  --package "$OUT/blind_preferences.jsonl" --mapping "$OUT/preferences_mapping.json"

# Собрать пары после выставления оценок судьёй
python "$RK2/code/rank.py" assemble --generation "$OUT/logs/preference_generation.json" \
  --package "$OUT/blind_preferences.jsonl" --mapping "$OUT/preferences_mapping.json" \
  --ratings "$OUT/preference_ratings.jsonl" --output "$OUT"

# Для повторения зафиксированного эксперимента используйте $RK2/data вместо $OUT/data
python "$RK2/code/train.py" --preference-dir "$OUT/data" --output "$OUT"
python "$RK2/code/predict.py" --output "$OUT" --batch-size 12
python "$RK2/code/evaluate.py" prepare --output "$OUT"

# Посчитать метрики после оценки новых тестовых ответов
python "$RK2/code/evaluate.py" aggregate --output "$OUT" --scores "$OUT/test_ratings.jsonl"
python "$RK2/code/plot.py" --output "$OUT"
```

Для разметки передайте судье слепой пакет и рубрики `RUBRIC` и `EDITOR_RUBRIC` из `rank.py`. Разметка preference содержит по строке с `id`, оценками вариантов в `scores` и необязательным исправлением `repair`. В `repair` фиксируются метка исходного варианта `source`, текст `answer`, три оценки и комментарий `note`. Для теста используйте рубрику из `evaluate.py` и JSONL вида `{"id":"...","A":{...},"B":{...}}`. Баллы по трём осям целые от 0 до 4.

Для пересчёта сохранённых результатов возьмите `ratings` и `repairs` из `logs/preference.json`. Каждый `id` образует строку `{"id": id, "scores": ratings[id]}`. Для записей с редактурой добавьте `"repair": repairs[id]`. Для теста запишите каждый элемент массива `ratings` из `logs/evaluation.json` отдельной строкой. Для повторной разметки используются GPT‑6.2 Sol и сохранённые рубрики.

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
| `data/prompt_pool_*.jsonl` | Вопросы и редакторские ориентиры для расширенного сбора |
| `data/test_fresh.jsonl` | 24 новых тестовых вопроса; 20 старых берутся из `hw_01_sft/data/test.jsonl` |
| `adapter/` | Итоговый LoRA-адаптер в `safetensors` и конфигурация PEFT |
| `logs/preference_generation.json`, `logs/preference.json` | Кандидаты, параметры, оценки судьи, отбор и исключения |
| `logs/training.json` | Окружение, гиперпараметры, loss, rewards и правило выбора |
| `logs/prediction.json`, `logs/evaluation.json` | Происхождение тестовых ответов и слепая разметка |
| `metrics/`, `plots/` | Ответы моделей, таблицы оценок и сравнения, графики |

Подключение сохранённого DPO использует `PeftModel.from_pretrained(base_model, 'hw_02_dpo/adapter')`. Полный пример загрузки базы и профили T4 NF4 и Apple Metal FP16 находятся в `code/model.py`. DPO-адаптер содержит параметры после продолжения SFT и подключается напрямую к базе.
