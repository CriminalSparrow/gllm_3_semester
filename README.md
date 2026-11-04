# ДЗ по генеративным LLM: автосервис

- [Отчёт](hw_01_sft/REPORT.md)
- [Ноутбук с выполненными ячейками](hw_01_sft/autoservice_sft.ipynb)
- [Запуск и описание файлов](hw_01_sft/README.md)
- [Данные SFT](hw_01_sft/data/)
- [Веса LoRA](hw_01_sft/adapter/adapter_model.safetensors)

Qwen2.5-1.5B-Instruct, QLoRA, Google Colab T4. Формат ответов: 0% → 100%; убедительного улучшения содержания не получено.

В репозитории находятся `.gitignore`, этот README и вся папка `hw_01_sft/`, включая веса и outputs ноутбука. Итоговый адаптер занимает около 35 МиБ и помещается в обычный Git. `.env`, `.venv`, кэши и промежуточные запуски исключены через `.gitignore`.
