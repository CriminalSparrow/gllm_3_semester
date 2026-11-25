# ДЗ по генеративным LLM: автосервис

## РК2 · DPO

- [Отчёт](hw_02_dpo/REPORT.md)
- [Ноутбук Colab](hw_02_dpo/autoservice_dpo.ipynb)
- [Артефакты и инструкция запуска](hw_02_dpo/README.md)

## РК1 · SFT

- [Отчёт](hw_01_sft/REPORT.md)
- [Ноутбук с выполненными ячейками](hw_01_sft/autoservice_sft.ipynb)
- [Запуск и описание файлов](hw_01_sft/README.md)
- [Данные SFT](hw_01_sft/data/)
- [Веса LoRA](hw_01_sft/adapter/adapter_model.safetensors)

Qwen2.5-1.5B-Instruct, QLoRA, Google Colab T4. Формат ответов: 0% → 100%; убедительного улучшения содержания не получено.

Папка `hw_01_sft/` включает веса и outputs ноутбука. Итоговый адаптер занимает около 35 МиБ. `.env`, `.venv`, кэши и промежуточные запуски исключены через `.gitignore`.
