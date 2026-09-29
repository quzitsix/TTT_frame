"""Model-free checks for the batch MCQ evaluator."""

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_spatial_items.py"
SPEC = importlib.util.spec_from_file_location("_ttt_frame_batch_mcq", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _item():
    return {
        "item_id": "x",
        "axis": "location",
        "question": "Where?",
        "options": {"A": "sink", "B": "island", "C": "drawer", "D": "fridge"},
        "answer": "B",
    }


def test_render_prompt_excludes_metadata_and_matches_native_format():
    prompt = module.render_prompt(_item())
    assert prompt == (
        "Where?\n\nA. sink\nB. island\nC. drawer\nD. fridge\n\n"
        "Answer with the single letter of the best option and nothing else."
    )
    assert "item_id" not in prompt
    assert "gold" not in prompt.casefold()


def test_score_and_summary_parse_letter_and_option_text():
    item = _item()
    scored = module.score_reply("B. island", item)
    assert scored["parsed"] == "B"
    assert scored["correct"] is True
    rows = [
        {"item_id": "x", "axis": "location", "correct": True},
        {"item_id": "y", "axis": "order", "correct": False},
    ]
    summary = module.summarize(rows)
    assert summary["correct"] == 1
    assert summary["n"] == 2
    assert summary["by_axis"]["location"]["accuracy"] == 1.0


def test_load_items_validates_four_options(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    assert module.load_items(path)[0]["item_id"] == "x"
