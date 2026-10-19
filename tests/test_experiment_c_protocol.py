from collections import Counter, defaultdict
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_experiment_c import CONDITIONS, condition_stream, event_records
from scripts.evaluate_experiment_c_visual import (
    _answer_pools,
    _choices,
    _parse_mcq_letters,
    _query_prompt,
)
from scripts.evaluate_experiment_c_jepa import _condition_episode, _training_indices
from scripts.prepare_experiment_c import build_episode
from scripts.summarize_experiment_c_visual import (
    _content_independence_groups,
    _metric,
    _paired_delta,
    _validate_against_manifest,
    summarize,
)


def _episodes():
    return [build_episode(index, seed=20261015, filler_before=4, filler_after=4)
            for index in range(12)]


def test_c_v2_withholds_future_and_matches_condition_budgets():
    episodes = _episodes()
    for episode in episodes:
        owner = next(query["answer"] for query in episode["queries"]
                     if query["type"] == "owner")
        assert episode["mover"] != owner
        assert all(event["kind"] != "forecast" for event in episode["events"])
        assert episode["future_event"]["kind"] == "forecast"
        for condition in CONDITIONS:
            events, answers = condition_stream(episode, condition)
            assert len(events) == 16
            assert sum(len(event_records(event)) for event in events) == 27
            assert all(event["kind"] != "forecast" for event in events)
            assert set(answers) == {query["id"] for query in episode["queries"]}


def test_c_v2_mcq_letters_are_balanced_per_query_type():
    episodes = _episodes()
    manifest = {"episodes": episodes}
    pools = _answer_pools(manifest, list(CONDITIONS))
    counts = defaultdict(Counter)
    for episode in episodes:
        for condition in CONDITIONS:
            _, answers = condition_stream(episode, condition)
            for query in episode["queries"]:
                _, letter = _choices(
                    query, answers[query["id"]], episode_id=episode["episode_id"],
                    condition=condition, pools=pools,
                )
                counts[query["type"]][letter] += 1
    assert counts
    assert all(counter == Counter({letter: 9 for letter in "ABCD"})
               for counter in counts.values())


def test_c_v2_mcq_parser_rejects_explanatory_prose():
    assert _parse_mcq_letters("A B C D A", 5) == list("ABCDA")
    assert _parse_mcq_letters('["A", "B", "C", "D", "A"]', 5) == list("ABCDA")
    assert _parse_mcq_letters("1. A\n2. B\n3. C\n4. D\n5. A", 5) == list("ABCDA")
    assert _parse_mcq_letters("Answers are A B C D A", 5) == []
    assert _parse_mcq_letters("A B C", 5) == []


def test_c_v2_duplicate_content_is_grouped_for_inference_and_probes():
    episodes = _episodes()
    groups = _content_independence_groups({"episodes": episodes})
    assert len(set(groups.values())) == 9
    assert groups["c-000"] == groups["c-006"]
    assert groups["c-001"] == groups["c-010"]
    assert groups["c-002"] == groups["c-008"]
    train = _training_indices([groups[episode["episode_id"]] for episode in episodes], 0)
    assert 0 not in train and 6 not in train


def test_jepa_legacy_forecast_filter_keeps_frame_paths_aligned():
    episode = build_episode(0, seed=20261015, filler_before=4, filler_after=4)
    episode["events"] = [*episode["events"], episode["future_event"]]
    episode["frame_paths"] = [f"frame-{index}.png" for index in range(len(episode["events"]))]
    conditioned = _condition_episode(episode, "relocated")
    assert len(conditioned["events"]) == 16
    assert len(conditioned["frame_paths"]) == 16
    assert all(event["kind"] != "forecast" for event in conditioned["events"])


def test_c_v2_paired_summary_requires_the_same_mcq_layout():
    def row(arm, query_id, correct, layout):
        return {
            "arm": arm, "episode_id": "c-000", "condition": "relocated",
            "query_id": query_id, "status": "ok", "mcq_layout": layout,
            "score": {"choice_correct": correct},
        }

    rows = [
        row("spatial", "q1", True, "separate"),
        row("spatial_base", "q1", False, "separate"),
        row("spatial", "q2", True, "separate"),
        row("spatial_base", "q2", False, "batched"),
    ]
    metric = _paired_delta(rows, "spatial", "spatial_base")
    assert metric["matched"] == 1
    assert metric["accuracy_delta"] == 1.0
    assert metric["left_only_correct"] == 1
    assert metric["right_only_correct"] == 0
    assert metric["exact_mcnemar_p"] == 1.0


def test_c_v2_summary_rebuilds_manifest_mcq_and_reparses_raw_response():
    episodes = _episodes()
    manifest = {"schema": "experiment_c_relocation_v2", "episodes": episodes}
    episode = episodes[0]
    condition = "relocated"
    query = episode["queries"][0]
    _, answers = condition_stream(episode, condition)
    gold = answers[query["id"]]
    pools = _answer_pools(manifest, list(CONDITIONS))
    options, gold_choice = _choices(
        query, gold, episode_id=episode["episode_id"], condition=condition, pools=pools,
    )
    row = {
        "arm": "icl", "episode_id": episode["episode_id"], "condition": condition,
        "query_id": query["id"], "query_type": query["type"], "gold": gold,
        "question": _query_prompt(
            query, forecast_hidden=True, condition=condition, answer_mode="mcq"
        ),
        "options": options, "gold_choice": gold_choice,
        "raw_response": gold_choice, "predicted_choice": gold_choice,
        "parse_valid": True, "mcq_layout": "separate", "observed_frames": 16,
        "run_protocol": {"max_frames": 0},
    }
    _validate_against_manifest([row], manifest)
    row["query_type"] = "owner" if query["type"] != "owner" else "mover"
    with pytest.raises(ValueError, match="manifest query type mismatch"):
        _validate_against_manifest([row], manifest)
    row["query_type"] = query["type"]
    row["raw_response"] = "The answer is " + gold_choice
    with pytest.raises(ValueError, match="raw response parse mismatch"):
        _validate_against_manifest([row], manifest)


def test_c_v2_summary_fails_closed_and_counts_errors(tmp_path):
    episode = _episodes()[0]
    condition = "relocated"
    _, answers = condition_stream(episode, condition)
    pools = _answer_pools({"episodes": _episodes()}, list(CONDITIONS))
    rows = []
    for query in episode["queries"]:
        gold = answers[query["id"]]
        options, gold_choice = _choices(
            query, gold, episode_id=episode["episode_id"], condition=condition, pools=pools,
        )
        rows.append({
            "episode_id": episode["episode_id"], "condition": condition, "arm": "icl",
            "query_id": query["id"], "query_type": query["type"],
            "question": query["question"], "gold": gold, "options": options,
            "gold_choice": gold_choice, "predicted_choice": gold_choice,
            "raw_response": gold_choice,
            "parse_valid": True, "status": "ok", "observed_frames": 16,
            "forecast_card_hidden": True, "answer_mode": "mcq", "mcq_layout": "separate",
            "card_protocol": "experiment_c_visual_v2",
            "episode_memory_state": "visual_context", "unique_observed_frames": 16,
            "condition_visual_encode_passes": 5,
            "condition_visual_frame_presentations": 80,
            "frames_provided_to_answer": 16,
            "score": {"choice_correct": True},
        })
    source = tmp_path / "rows.jsonl"
    source.write_text("placeholder")
    with pytest.raises(ValueError, match="run_protocol"):
        summarize(
            rows, [source], expected_episodes=1,
            expected_conditions={"relocated"}, expected_observed_frames=16,
            expected_arms={"icl"},
        )
    rows[0]["status"] = "error"
    with pytest.raises(ValueError, match="error rows cannot contain"):
        summarize(
            rows, [source], allow_legacy_metadata=True, expected_episodes=1,
            expected_conditions={"relocated"}, expected_observed_frames=16,
            expected_arms={"icl"},
        )
    rows[0]["status"] = "ok"
    summary = summarize(
        rows, [source], allow_legacy_metadata=True, expected_episodes=1,
        expected_conditions={"relocated"}, expected_observed_frames=16,
        expected_arms={"icl"},
    )
    assert summary["protocol_validation"] == "passed"
    assert summary["chance_accuracy"] == 0.25
    assert summary["legacy_metadata_override"]

    rows[0]["forecast_card_hidden"] = False
    with pytest.raises(ValueError, match="future-visible"):
        summarize(
            rows, [source], allow_legacy_metadata=True, expected_episodes=1,
            expected_conditions={"relocated"}, expected_observed_frames=16,
            expected_arms={"icl"},
        )
    rows[0]["forecast_card_hidden"] = True
    for row in rows:
        row["run_protocol"] = {}
    with pytest.raises(ValueError, match="fixture schema v2"):
        summarize(
            rows, [source], expected_episodes=1,
            expected_conditions={"relocated"}, expected_observed_frames=16,
            expected_arms={"icl"},
        )

    metric_rows = [
        {"status": "ok", "parse_valid": True, "score": {"choice_correct": True}},
        {"status": "error", "parse_valid": False, "score": {"choice_correct": False}},
    ]
    assert _metric(metric_rows)["choice_accuracy"] == 0.5
    assert _metric(metric_rows)["parse_rate"] == 0.5
