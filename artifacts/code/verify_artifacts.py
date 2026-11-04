"""Проверка сдаваемого комплекта без GPU и без обращения к API."""
import hashlib
import json
import ast
from pathlib import Path
import sys
from common import ROOT, check_answer, check_examples, read_jsonl, read_log, model_check_examples

def main():
    check_examples()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for filename, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / "data" / filename).read_bytes()).hexdigest() == expected
    questions = set()
    for split, n in [("train", 112), ("validation", 16)]:
        rows = read_jsonl(ROOT / f"data/{split}.jsonl")
        assert len(rows) == n
        for r in rows:
            assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
            assert check_answer(r["messages"][-1]["content"])["pass"]
            q = r["messages"][1]["content"].strip().lower()
            assert q not in questions
            questions.add(q)
    test = read_jsonl(ROOT / "data/test.jsonl")
    assert len(test) == 20
    assert not questions & {r["question"].strip().lower() for r in test}
    for stage in ["before", "after"]:
        rows = read_jsonl(ROOT / f"metrics/{stage}_answers.jsonl")
        assert [r["id"] for r in rows] == [r["id"] for r in test]
        for r in rows:
            measured = check_answer(r["answer"])
            assert all(r[k] == measured[k] for k in measured)
    import nbformat
    notebook = nbformat.read(ROOT / "autoservice_sft.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(isinstance(cell.execution_count, int) and cell.execution_count > 0 for cell in code_cells)
    assert all(not any(output.output_type == "error" for output in cell.outputs) for cell in code_cells)
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    assert all(cell.outputs for cell in code_cells)
    for index, cell in enumerate(code_cells[1:], 1):
        imports = [node for node in ast.walk(ast.parse(cell.source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert bool(imports) == (index == 1), f"Импорты в ячейке {index}"
    assert "RUN_TRAINING = True" in "\n".join(cell.source for cell in code_cells)
    assert read_jsonl(ROOT / "metrics/check_examples.jsonl") == model_check_examples()
    import numpy as np
    from safetensors import safe_open
    with safe_open(ROOT / "adapter/adapter_model.safetensors", framework="np") as f:
        keys = list(f.keys())
        assert keys and all("lora_" in k for k in keys)
        assert all(np.isfinite(f.get_tensor(k)).all() for k in keys)
        assert any(np.any(f.get_tensor(k)) for k in keys if "lora_B" in k)
        print("Adapter:", len(keys), "tensors, finite, learned B != 0")
    assert len(read_log("judge")["evaluations"]) == 40
    assert len(read_log("data_preparation")["generations"]) == 16
    assert len(read_log("data_preparation")["editorial_edits"]) == 24
    assert {path.name for path in (ROOT / "logs").iterdir()} == {"data_preparation.json", "training.json", "judge.json"}
    assert (ROOT / "plots/comparison.png").is_file()
    assert (ROOT / "plots/loss.png").is_file()
    assert len((ROOT / "REPORT.md").read_text()) > 10000
    print(f"Notebook: {len(code_cells)} code cells executed, no errors; imports after pip: OK")
    print("Data, notebook, metrics, adapter, 3 JSON logs and report: OK")

if __name__ == "__main__":
    main()
