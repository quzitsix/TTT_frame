#!/usr/bin/env python3
"""Prepare a small, trajectory-preserving SVCBench subset.

SVCBench has one row per streaming query point in ``vcbench_eval.jsonl``.
This helper groups those rows back into questions and source videos so that a
runner can read one video once and evaluate the complete count trajectory.
It does not download media.

The default selection is the generated ``simulated/0030.mp4`` video.  It is a
small, redistributable source video with two questions and 47 query points.

Example::

    python scripts/prepare_svcbench_subset.py \
      --annotation /data/svcbench/vcbench_eval.jsonl \
      --output /data/svcbench/subset \
      --video simulated/0030.mp4
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_VIDEO = "simulated/0030.mp4"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("every annotation line must be a JSON object")
                rows.append(item)
    if not rows:
        raise ValueError(f"annotation file is empty: {path}")
    return rows


def _video_key(row: dict[str, Any]) -> str:
    source = str(row.get("source_dataset", "")).strip()
    video = str(row.get("video_path", "")).strip()
    if not source or not video:
        raise ValueError(f"annotation row has no source_dataset/video_path: {row}")
    return f"{source}/{video.lstrip('/')}"


def _normalize_subtype(value: Any) -> str:
    """Keep the published spelling, while exposing a stable coarse subtype."""

    value = str(value or "Unknown")
    aliases = {
        "O1-Snapshot": "O1-Snap",
        "E1-Act": "E1-Action",
        "E1-Trans": "E1-Transit",
        "E2-Episodic": "E2-Episode",
        "E2-Cyclic": "E2-Periodic",
    }
    return aliases.get(value, value)


def group_rows(rows: Iterable[dict[str, Any]], videos: set[str]) -> list[dict[str, Any]]:
    """Return grouped question trajectories for the requested source videos."""

    selected = [row for row in rows if _video_key(row) in videos]
    if not selected:
        available = sorted({_video_key(row) for row in rows})
        raise ValueError(
            "none of the requested videos occur in the annotation; "
            f"requested={sorted(videos)}, examples={available[:8]}"
        )

    groups: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
    for row in selected:
        key = (_video_key(row), str(row.get("id", row.get("q_id", ""))))
        if key not in groups:
            groups[key] = {
                "id": key[1],
                "source_dataset": row["source_dataset"],
                "video_path": row["video_path"],
                "question": row.get("question", ""),
                "counting_type": row.get("counting_type", ""),
                "counting_subtype": row.get("counting_subtype", "Unknown"),
                "counting_subtype_normalized": _normalize_subtype(row.get("counting_subtype")),
                "query_points": [],
            }
        groups[key]["query_points"].append({
            "q_id": str(row.get("q_id", "")),
            "query_index": int(row.get("query_index", len(groups[key]["query_points"]))),
            "query_time": float(row["query_time"]),
            "count": int(row["count"]),
        })

    result = list(groups.values())
    for item in result:
        item["query_points"].sort(key=lambda point: (point["query_time"], point["query_index"]))
        item["query_times"] = [point["query_time"] for point in item["query_points"]]
        item["counts"] = [point["count"] for point in item["query_points"]]
        item["n_query_points"] = len(item["query_points"])
        item["video_key"] = f"{item['source_dataset']}/{item['video_path']}"
    return result


def prepare(annotation: Path, output: Path, videos: list[str]) -> dict[str, Any]:
    rows = _load_rows(annotation)
    requested = set(videos)
    grouped = group_rows(rows, requested)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ttt_frame.svcbench_subset/1",
        "source_annotation": str(annotation.resolve()),
        "requested_videos": sorted(requested),
        "videos": [
            {
                **item,
                "video_file": f"videos/{item['source_dataset']}/{item['video_path']}",
            }
            for item in grouped
        ],
        "n_source_rows": len(rows),
        "n_questions": len(grouped),
        "n_query_points": sum(item["n_query_points"] for item in grouped),
    }
    subset_path = output / "svcbench_subset.json"
    subset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # A manifest is convenient for a selective Hugging Face download and is
    # intentionally one path per source video, rather than one per query.
    manifest = sorted({f"videos/{item['source_dataset']}/{item['video_path']}" for item in grouped})
    (output / "download_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    # Also retain the flat rows for tools that already consume vcbench_eval's
    # one-row-per-query format.
    flat = []
    for item in grouped:
        for point in item["query_points"]:
            flat.append({
                "q_id": point["q_id"],
                "id": item["id"],
                "source_dataset": item["source_dataset"],
                "video_path": item["video_path"],
                "question": item["question"],
                "counting_type": item["counting_type"],
                "counting_subtype": item["counting_subtype"],
                "query_time": point["query_time"],
                "count": point["count"],
                "query_index": point["query_index"],
            })
    with (output / "svcbench_eval_subset.jsonl").open("w", encoding="utf-8") as handle:
        for row in flat:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "annotation": str(annotation.resolve()),
        "output": str(output.resolve()),
        "videos": manifest,
        "n_source_rows": len(rows),
        "n_questions": len(grouped),
        "n_query_points": len(flat),
        "questions": [
            {
                "id": item["id"],
                "video": item["video_key"],
                "counting_subtype": item["counting_subtype_normalized"],
                "n_query_points": item["n_query_points"],
                "first_query_time": item["query_times"][0],
                "last_query_time": item["query_times"][-1],
            }
            for item in grouped
        ],
    }
    (output / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="append", dest="videos", default=None,
                        help="source/path.mp4; repeat for multiple videos (default: simulated/0030.mp4)")
    args = parser.parse_args(argv)
    if not args.annotation.is_file():
        raise SystemExit(f"annotation does not exist: {args.annotation}")
    videos = args.videos or [DEFAULT_VIDEO]
    print(json.dumps(prepare(args.annotation, args.output, videos), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
