#!/usr/bin/env python3
"""Select a small, reproducible FHO-STA validation subset.

The Ego4D annotations are gated and the complete video download is several
terabytes.  This helper never downloads media.  It reads an already downloaded
``fho_sta_val.json``, selects a balanced number of annotations by time to
contact, writes a compact annotation file, and emits UID lists for the Ego4D
CLI.  The lists can then be used to download only the corresponding FHO clips.

Example::

    python scripts/prepare_fho_sta_subset.py \
      --annotation /data/quzitsix/ego4d/v2_1/annotations/fho_sta_val.json \
      --output runs/fho_sta_subset --limit 16 --seed 0

The generated ``clip_uids.txt`` and ``video_uids.txt`` are whitespace-delimited
files accepted by ``ego4d --video_uid_file``.  For the ``clips`` dataset use
``clip_uids.txt``; for ``full_scale`` use ``video_uids.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


HORIZON_BINS = (0.5, 1.0, 1.5, 2.0)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        raise ValueError("expected a FHO annotation object with an annotations list")
    return payload


def _objects(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    objects = annotation.get("objects", [])
    return [obj for obj in objects if isinstance(obj, dict) and isinstance(
        obj.get("time_to_contact"), (int, float)
    ) and float(obj["time_to_contact"]) >= 0]


def _bin(value: float) -> float:
    return min(HORIZON_BINS, key=lambda target: abs(target - value))


def _annotation_key(annotation: dict[str, Any]) -> str:
    uid = annotation.get("uid")
    if uid is None:
        raise ValueError("annotation is missing uid")
    return str(uid)


def select_annotations(payload: dict[str, Any], *, limit: int, seed: int) -> list[dict[str, Any]]:
    """Select at most ``limit`` items while covering TTC bins when possible."""

    if limit < 1:
        raise ValueError("limit must be positive")
    candidates = [ann for ann in payload["annotations"] if _objects(ann)]
    if not candidates:
        raise ValueError("no labeled annotations with time_to_contact were found")
    rng = random.Random(seed)
    groups: dict[float, list[dict[str, Any]]] = {target: [] for target in HORIZON_BINS}
    for annotation in candidates:
        ttc = min(float(obj["time_to_contact"]) for obj in _objects(annotation))
        groups[_bin(ttc)].append(annotation)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    # Round-robin keeps a small smoke run useful at every short-term horizon.
    while len(selected) < min(limit, len(candidates)):
        added = False
        for target in HORIZON_BINS:
            if groups[target] and len(selected) < limit:
                selected.append(groups[target].pop())
                added = True
        if not added:
            break
    selected.sort(key=lambda ann: (_annotation_key(ann)))
    return selected


def _copy_payload(payload: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(payload)
    result["annotations"] = selected
    # Keep the schema/version visible in the subset so a later evaluator cannot
    # accidentally combine annotations from another Ego4D release.
    result["subset_metadata"] = {
        "source_annotation_count": len(payload["annotations"]),
        "selected_annotation_count": len(selected),
        "selection": "round_robin_time_to_contact",
    }
    return result


def prepare(annotation: Path, output: Path, *, limit: int, seed: int) -> dict[str, Any]:
    payload = _load(annotation)
    selected = select_annotations(payload, limit=limit, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fho_sta_val_subset.json").write_text(
        json.dumps(_copy_payload(payload, selected), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    clip_uids = sorted({str(ann["clip_uid"]) for ann in selected if ann.get("clip_uid")})
    video_uids = sorted({str(ann.get("video_uid", ann.get("video_id"))) for ann in selected
                         if ann.get("video_uid", ann.get("video_id"))})
    (output / "clip_uids.txt").write_text("\n".join(clip_uids) + "\n", encoding="utf-8")
    (output / "video_uids.txt").write_text("\n".join(video_uids) + "\n", encoding="utf-8")

    report = {
        "source_annotation": str(annotation.resolve()),
        "output": str(output.resolve()),
        "seed": seed,
        "selected": len(selected),
        "clip_uids": len(clip_uids),
        "video_uids": len(video_uids),
        "horizon_bins": {
            str(target): sum(
                _bin(min(float(obj["time_to_contact"]) for obj in _objects(ann))) == target
                for ann in selected
            )
            for target in HORIZON_BINS
        },
        "files": {
            "annotations": str((output / "fho_sta_val_subset.json").resolve()),
            "clip_uids": str((output / "clip_uids.txt").resolve()),
            "video_uids": str((output / "video_uids.txt").resolve()),
        },
    }
    (output / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.annotation.is_file():
        raise SystemExit(f"annotation does not exist: {args.annotation}")
    print(json.dumps(prepare(args.annotation, args.output, limit=args.limit, seed=args.seed),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
