"""Small, model-free checks for the original-item test CLI."""

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "test_spatial_memory.py"
SPEC = importlib.util.spec_from_file_location("_ttt_frame_spatial_memory_cli_impl", SCRIPT)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


def test_render_item_prompt_sorts_options_and_requires_one_letter():
    prompt = cli.render_item_prompt(
        {
            "answer_format": "mcq",
            "question": "Where?",
            "options": {"B": "second", "A": "first"},
            "answer": "A",
        }
    )
    assert prompt == (
        "Where?\n\nA. first\nB. second\n\n"
        "Answer with the single letter of the best option and nothing else."
    )
    assert "Gold:" not in prompt


def test_load_item_reads_one_jsonl_row(tmp_path):
    path = tmp_path / "items.jsonl"
    rows = [{"item_id": "one", "question": "Q", "answer": "B"}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert cli.load_item(path, "one") == rows[0]
