import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_homesentinel_temporal_pilot import main  # noqa: E402
from ttt_frame.predictive_dataset import (  # noqa: E402
    build_evidence_index,
    build_homesentinel_temporal_manifest,
    parse_time_seconds,
)


def _write_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "asuka"
    video_id = "source-with-hyphens-1"
    video_root = root / video_id
    (video_root / "caption").mkdir(parents=True)
    (video_root / "raw_video.mp4").write_bytes(b"not decoded by candidate mining")
    (video_root / "dataset_meta.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "valid_scenes": [
                    {"scene_id": "scene_001", "start": 0.0, "end": 10.0},
                    {"scene_id": "scene_002", "start": 20.0, "end": 30.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "merged_captions.json").write_text(
        json.dumps(
            [
                {
                    "video_id": video_id,
                    "video_idx": 1,
                    "segments": [
                        {
                            "scene_id": "scene_001",
                            "segment_id": "seg_extracted",
                            "events": [{"t": "00:02", "text": "automatic prose"},
                                       {"t": "00:05", "text": "automatic prose"}],
                        },
                        {
                            "scene_id": "scene_002",
                            "segment_id": "seg_extracted",
                            "events": [{"t": "00:22", "text": "automatic prose"},
                                       {"t": "00:25", "text": "automatic prose"}],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    objects = root / "benchmark_tmp"
    objects.mkdir()
    first = f"{video_id}-scene_001-seg_extracted"
    second = f"{video_id}-scene_002-seg_extracted"
    (objects / "video_1_object.json").write_text(
        json.dumps(
            [
                {
                    "object": "striped mug",
                    "observations": [
                        {"location/state": "on table", "video_idx": 1, "timestamp": first},
                        {"location/state": "in cabinet", "video_idx": 1, "timestamp": second},
                    ],
                },
                {
                    "object": "broken reference",
                    "observations": [
                        {"location/state": "somewhere", "video_idx": 1,
                         "timestamp": f"{video_id}-scene_999-seg_extracted"},
                        {"location/state": "elsewhere", "video_idx": 1, "timestamp": second},
                    ],
                },
                {
                    "object": "unchanged item",
                    "observations": [
                        {"location/state": "on shelf", "video_idx": 1, "timestamp": first},
                        {"location/state": "on shelf", "video_idx": 1, "timestamp": second},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    return root


def test_parse_time_seconds():
    assert parse_time_seconds("01:02") == 62
    assert parse_time_seconds("01:02:03") == 3723
    with pytest.raises(ValueError):
        parse_time_seconds("1:99")


def test_evidence_index_handles_hyphenated_video_id_and_real_scene_ranges(tmp_path):
    root = _write_fixture(tmp_path)
    index, issues = build_evidence_index(root)
    evidence_id = "source-with-hyphens-1-scene_002-seg_extracted"
    assert evidence_id in index
    assert index[evidence_id]["video_id"] == "source-with-hyphens-1"
    assert index[evidence_id]["media_scene_range_seconds"] == {
        "start": 20.0,
        "end": 30.0,
        "source": "dataset_meta.valid_scenes",
    }
    assert index[evidence_id]["caption_event_span_seconds"]["start"] == 22.0
    assert not issues


def test_manifest_is_candidate_queue_not_automatic_gold(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = build_homesentinel_temporal_manifest(root, min_gap_seconds=10)

    assert manifest["human_verified"] is False
    assert manifest["label_policy"]["candidate_mining_only"] is True
    assert manifest["label_policy"]["automatic_caption_is_gold"] is False
    assert len(manifest["candidates"]) == 1
    candidate = manifest["candidates"][0]
    assert candidate["object"] == "striped mug"
    assert candidate["gap_seconds"] == 17.0
    assert [row["location_state_text"] for row in candidate["ordered_observations"]] == [
        "on table",
        "in cabinet",
    ]
    assert candidate["human_annotation"]["answer"] is None
    assert candidate["suggested_tasks"]["C2_latest_state"]["human_answer"] is None
    assert candidate["suggested_tasks"]["C3_delayed_binding_revisit"]["human_answer"] is None
    assert manifest["summary"]["unresolved_references"] == 1
    assert any(
        row["reason"] == "state_text_not_distinct"
        for row in manifest["diagnostics"]["skipped"]
    )


def test_cli_writes_configurable_limited_manifest(tmp_path):
    root = _write_fixture(tmp_path)
    output = tmp_path / "pilot.json"
    assert main([
        "--data-root", str(root),
        "--output", str(output),
        "--min-gap", "5",
        "--limit", "1",
    ]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["config"] == {"min_gap_seconds": 5.0, "limit": 1}
    assert manifest["summary"]["candidates_written"] == 1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"min_gap_seconds": float("nan")}, "finite nonnegative"),
        ({"limit": True}, "nonnegative integer"),
        ({"objects_dir": Path("missing")}, "objects directory"),
    ],
)
def test_manifest_rejects_invalid_configuration(tmp_path, options, message):
    root = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match=message):
        build_homesentinel_temporal_manifest(root, **options)
