import csv
import json

from rich.console import Console

from docetl.dataset import DataLoader
from docetl.runner import save_output
from docetl.utils import extract_output_from_json, load_config


UNICODE_TEXT = "你好 € café"


def test_json_dataset_reads_utf8_bom(tmp_path):
    path = tmp_path / "input.json"
    records = [{"text": UNICODE_TEXT}]
    path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8-sig",
    )

    loader = DataLoader(None, "file", str(path))

    assert loader.load() == records


def test_csv_dataset_reads_utf8_bom_and_counts_rows(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(f"text\n{UNICODE_TEXT}\n", encoding="utf-8-sig")
    loader = DataLoader(None, "file", str(path))

    assert loader.load() == [{"text": UNICODE_TEXT}]
    assert loader.count() == 1


def test_load_config_reads_utf8_bom(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        f'default_model: gpt-4o-mini\nnote: "{UNICODE_TEXT}"\n',
        encoding="utf-8-sig",
    )

    assert load_config(str(path))["note"] == UNICODE_TEXT


def test_extract_output_reads_utf8_bom(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    output_path = tmp_path / "output.json"
    records = [{"text": UNICODE_TEXT}]
    config_path.write_text(
        f'note: "{UNICODE_TEXT}"\n'
        "pipeline:\n"
        "  steps:\n"
        "    - operations: [final]\n"
        "operations:\n"
        "  - name: final\n"
        "    output:\n"
        "      schema:\n"
        "        text: string\n",
        encoding="utf-8-sig",
    )
    output_path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8-sig",
    )

    assert extract_output_from_json(str(config_path), str(output_path)) == records


def test_save_output_writes_unicode_as_utf8(tmp_path):
    records = [{"text": UNICODE_TEXT}]
    console = Console(quiet=True)
    json_path = tmp_path / "output.json"
    csv_path = tmp_path / "output.csv"

    save_output(records, str(json_path), console)
    save_output(records, str(csv_path), console)

    json_text = json_path.read_text(encoding="utf-8")
    assert UNICODE_TEXT in json_text
    assert json.loads(json_text) == records
    with csv_path.open(encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == records
