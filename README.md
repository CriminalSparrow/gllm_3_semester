# ДЗ по генеративным LLM: автосервис

- [Отчёт](artifacts/REPORT.md)
- [Ноутбук с выполненными ячейками](artifacts/autoservice_sft.ipynb)
- [Запуск и описание файлов](artifacts/README.md)
- [Данные SFT](artifacts/data/)
- [Веса LoRA](artifacts/adapter/adapter_model.safetensors)

Qwen2.5-1.5B-Instruct, QLoRA, Google Colab T4. Формат ответов: 0% → 100%; убедительного улучшения содержания не получено.

В репозитории находятся `.gitignore`, этот README и вся папка `artifacts/`, включая веса и outputs ноутбука. Итоговый адаптер занимает около 35 МиБ и помещается в обычный Git. `.env`, `.venv`, кэши и промежуточные запуски исключены через `.gitignore`.
