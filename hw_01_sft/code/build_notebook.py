"""Собрать ноутбук. После сборки его нужно выполнить, outputs не подделываются."""
import base64
import io
import json
import zipfile
from common import ROOT


def bootstrap():
    buffer = io.BytesIO()
    paths = [ROOT / 'requirements.txt', ROOT / 'config.json']
    for folder in ['code', 'data', 'metrics', 'logs', 'plots']:
        paths += [p for p in (ROOT / folder).glob('*') if p.is_file()]
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, 'hw_01_sft/' + str(path.relative_to(ROOT)))
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    cells = []
    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text))
    def code(text):
        cells.append(dict(cell_type='code', metadata={}, source=text, execution_count=None, outputs=[]))

    md('# ДЗ: консультант автосервиса\n\nQwen2.5-1.5B-Instruct → QLoRA. '
       'Выберите **Google Colab → T4** и выполните ячейки сверху вниз. '
       'Данные, код и сохранённые оценки судьи встроены в ноутбук. '
       'Повторное дообучение сохраняется отдельно в `review_run/`. '
       'Таблицы отчёта используют зафиксированные ответы первого прогона и их оценки судьи. '
       'Отчёт — `REPORT.md`, инструкции — `README.md`.')
    code('!nvidia-smi\n%pip -q install transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 '
         'bitsandbytes==0.50.2 datasets==3.5.0 matplotlib==3.10.1')
    code('''# Все импорты ноутбука находятся в этой ячейке после pip install.
import base64
import getpass
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from IPython.display import Image, Markdown, display
from safetensors import safe_open
try:
    from google.colab import files
except ImportError:
    files = None

# Встроенный архив содержит исходники, данные и фактические оценки первого прогона.
PAYLOAD = ''' + repr(bootstrap()) + '''
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall('.')
ROOT = pathlib.Path('hw_01_sft').resolve()
sys.path.insert(0, str(ROOT / 'code'))
from common import read_jsonl, read_log, save_log, write_jsonl, check_answer, check_examples, model_check_examples, display_answer

for folder in ['logs', 'metrics', 'plots', 'adapter']:
    (ROOT / folder).mkdir(exist_ok=True)
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Выберите GPU T4'
print('GPU:', torch.cuda.get_device_name(0))
print('Файлы:', ROOT)
''')
    md('## 1. Данные\n112 диалогов для SFT, 16 для валидации, 20 отдельных тестовых вопросов. '
       'Ответы GLM-5.3-Flash; 24 опасных ответа исправлены до обучения.')
    code('''dataset = load_dataset('json', data_files={
    'train': str(ROOT/'data/train.jsonl'), 'validation': str(ROOT/'data/validation.jsonl')})
print(dataset)
print('Тестовых вопросов:', len(read_jsonl(ROOT/'data/test.jsonl')))
example = dataset['train'][0]['messages']
display(Markdown('**Вопрос:** ' + example[1]['content'] + '\\n\\n' + display_answer(example[2]['content'])))
''')
    md('## 2. Автоматическая проверка\n20–150 слов и четыре непустых раздела по порядку. '
       'Ниже два реальных отказа базы и два прошедших ответа адаптера из сохранённого прогона. '
       'Проверка оценивает формат; техническая правильность оценивается отдельно.')
    code('''assert len(check_examples()) == 6
examples = model_check_examples()
write_jsonl(ROOT/'metrics/check_examples.jsonl', examples)
for row in examples:
    verdict = 'прошёл' if row['pass'] else 'не прошёл'
    reason = 'все разделы на месте' if row['pass'] else 'отсутствуют обязательные разделы'
    display(Markdown(f"**{row['stage']}, {row['id']}: {verdict}, {row['words']} слов. {reason}.**\\n\\n"
                     + '**Вопрос:** ' + row['question'] + '\\n\\n' + display_answer(row['answer'])))
print('Шесть проверок граничных случаев: OK')
''')
    md('## 3. Базовая модель → SFT → оценка адаптера\nДве эпохи QLoRA, r=8, batch=2, накопление=4. '
       'Loss считается по ответу ассистента. Скрипт сначала получает базовые ответы, затем обучает модель, '
       'сохраняет адаптер и оценивает его после повторной загрузки с диска. '
       'Повторный запуск помещается в `review_run/`, чтобы сохранить исходные ответы и оценки отчёта. '
       'Из-за вычислений на GPU отдельные формулировки могут немного отличаться.')
    code('''RUN_TRAINING = True
if RUN_TRAINING:
    REPRO = ROOT/'review_run'
    for folder in ['code', 'data']:
        shutil.copytree(ROOT/folder, REPRO/folder, dirs_exist_ok=True)
    shutil.copy2(ROOT/'config.json', REPRO/'config.json')
    result = subprocess.run([sys.executable, '-u', str(REPRO/'code/run_experiment.py')],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    save_log('training', 'notebook_reproduction', {
        'console': result.stdout, 'returncode': result.returncode,
        'metrics': json.loads((REPRO/'logs/training.json').read_text()).get('metrics', {})})
    print(result.stdout)
    assert result.returncode == 0, 'Обучение завершилось ошибкой; см. logs/training.json → notebook_reproduction'
else:
    print('Повторное обучение выключено. Ниже сохранённый фактический прогон:')
    print(json.dumps(read_log('training')['metrics'], ensure_ascii=False, indent=2))
''')
    code('''for stage in ['before', 'after']:
    print('\\n' + stage.upper())
    for row in read_jsonl(ROOT/f'metrics/{stage}_answers.jsonl')[:2]:
        display(Markdown('**Вопрос:** ' + row['question'] + '\\n\\n' + display_answer(row['answer'])))
print((ROOT/'metrics/automatic_summary.json').read_text())
write_jsonl(ROOT/'metrics/check_examples.jsonl', model_check_examples())
''')
    md('## 4. LLM-as-judge\nGLM-5.3-Flash оценивает безопасность, корректность и полезность (0–4). '
       'Каждая пара оценивается в обоих порядках A/B. По умолчанию используются реальные оценки '
       'из первого API-прогона: код проверяет, что они относятся к сохранённым ответам отчёта. '
       'Для новых ответов включите `RUN_JUDGE=True` и введите ключ скрыто.')
    code('''RUN_JUDGE = False
if RUN_JUDGE:
    os.environ['LLM_URL'] = input('OpenAI-compatible base URL: ').strip()
    os.environ['LLM_TOKEN'] = getpass.getpass('API token: ')
    os.environ['JUDGE_MODEL'] = 'glm-4.7'
else:
    before = read_jsonl(ROOT/'metrics/before_answers.jsonl')
    after = read_jsonl(ROOT/'metrics/after_answers.jsonl')
    cached = read_log('judge')['evaluations']
    for left, right in zip(before, after):
        for order in [0, 1]:
            record = cached[f"{left['id']}_{order}"]
            answers = {'before': left['answer'], 'after': right['answer']}
            assert record['request'] == dict(question=left['question'], reference=left['reference'],
                A=answers[record['mapping']['A']], B=answers[record['mapping']['B']]), 'Ответы изменились: включите RUN_JUDGE=True'
    print('Использованы 40 сохранённых оценок API; вопросы и ответы совпадают.')
try:
    result = subprocess.run([sys.executable, '-u', str(ROOT/'code/judge.py')],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    save_log('judge', 'console', {'notebook': result.stdout})
    print(result.stdout)
    assert result.returncode == 0
finally:
    os.environ.pop('LLM_TOKEN', None)
''')
    md('## 5. Таблица, графики и вывод\nФормат стал стабильным, но убедительного улучшения содержания не получено. '
       'Все ответы и индивидуальные баллы сохранены в `metrics/`, сырые оценки — в `logs/judge.json`.')
    code('''subprocess.run([sys.executable, str(ROOT/'code/plot_results.py')], check=True)
display(pd.read_csv(ROOT/'metrics/summary.csv'))
display(Image(filename=str(ROOT/'plots/comparison.png')))
display(Image(filename=str(ROOT/'plots/loss.png')))
print('Победы:', json.loads((ROOT/'metrics/judge_summary.json').read_text())['wins'])
''')
    code('''adapter_path = (REPRO if RUN_TRAINING else ROOT)/'adapter/adapter_model.safetensors'
with safe_open(adapter_path, framework='np') as adapter:
    keys = list(adapter.keys())
    assert all(np.isfinite(adapter.get_tensor(key)).all() for key in keys)
    assert any(np.any(adapter.get_tensor(key)) for key in keys if 'lora_B' in key)
print('LoRA:', len(keys), 'тензоров; веса корректны, матрицы B обучены.')
print('Проверенный адаптер:', adapter_path)
print('Логи:', ', '.join(p.name for p in sorted((ROOT/'logs').glob('*.json'))))
''')
    md('## 6. Сохранение\nАрхив содержит данные, код, веса и результаты текущего запуска. '
       'В сдаваемом комплекте также есть README и отредактированный отчёт. '
       'Скачивание в Colab можно включить флагом ниже.')
    code('''archive = shutil.make_archive('autoservice_results', 'zip', root_dir='.', base_dir='hw_01_sft')
print('Архив создан:', archive, '; размер:', round(pathlib.Path(archive).stat().st_size/1024**2, 1), 'МиБ')
DOWNLOAD = False
if DOWNLOAD and files is not None:
    files.download(archive)
''')
    for i, cell in enumerate(cells):
        cell['id'] = f'autoservice-{i:02d}'
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': [], 'private_outputs': False},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'}})
    path = ROOT/'autoservice_sft.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print('Собран ноутбук без outputs. Теперь выполните все ячейки:', path)

if __name__ == '__main__':
    main()
