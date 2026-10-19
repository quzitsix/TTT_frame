#!/usr/bin/env python3
"""Run the controlled relocation/scene-change experiment C.

This is intentionally a *mechanism* runner rather than a VLM benchmark.  A
stable random semantic code is used in place of an image encoder, so the only
way to answer a question after observations are revoked is to retain the
key/value association.  The same event stream is evaluated with:

``oracle``
    the structured state table (an upper bound);
``explicit_retrieval``
    a small nearest-key table with the same fixed-size budget;
``lact``
    the repository's corrected LaCT fast-weight memory;
``none``
    no writes and no retained state (coverage is zero).

The ``relocated`` and ``same_scene_control`` conditions share the same event
and write budgets.  In fixture schema v2, the same-scene transform also maps
two room-specific location names to shorter scene-A names, so their difference
is a matched relocation-condition package rather than a pure scene-cue effect.
``filler_only_control`` replaces the target move with unrelated writes under
the same budget, so it measures ordinary retention rather than a state update.
``--scene-drift-strength 0`` is the invariant-object control; a positive value
tests whether a scene change damages the retrieval cue.  No network or model
download is required.

V2 stores the future ``forecast`` record as a sidecar target rather than an
observed event.  The evaluator also filters it from legacy manifests.
``next_action`` is composed from the retained moved-status, owner and
final-location bindings under the public policy "an owner checks the object's
final location after a move".  It is therefore a state-conditioned prediction
rather than recall of a supplied future label.

Examples
--------
    python scripts/prepare_experiment_c.py --output-dir data/experiment_c
    conda run --no-capture-output -n meowbench python scripts/evaluate_experiment_c.py \
      --data data/experiment_c/manifest.json \
      --output runs/experiment_c --trials 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttt_frame.lact import LaCTMemory  # noqa: E402


CONDITIONS = ("relocated", "same_scene_control", "filler_only_control")
ARMS = ("oracle", "explicit_retrieval", "lact", "none")
QUERY_TYPES = ("location", "mover", "owner", "state", "next_action")


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector.float(), dim=-1)


class SemanticCodebook:
    """Stable semantic vectors; avoids Python's process-randomized hash()."""

    def __init__(self, dim: int, seed: int) -> None:
        self.dim = dim
        self.seed = seed
        self._cache: dict[str, torch.Tensor] = {}

    def code(self, text: str) -> torch.Tensor:
        key = str(text)
        if key not in self._cache:
            digest = hashlib.sha256(f"{self.seed}/{key}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
            self._cache[key] = _unit(torch.randn(self.dim, generator=generator))
        return self._cache[key]

    def key(self, relation: str, obj: str, scene: str, drift: float) -> torch.Tensor:
        base = self.code(f"key/{relation}/{obj}")
        if not drift:
            return base
        return _unit(base + float(drift) * self.code(f"scene/{scene}"))

    def value(self, relation: str, value: str) -> torch.Tensor:
        return self.code(f"value/{relation}/{value}")


class ExplicitRetrieval:
    """A transparent fixed-size key/value table, used as a strong control."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self.entries: list[tuple[torch.Tensor, torch.Tensor, str]] = []

    def write(self, key: torch.Tensor, value: torch.Tensor, label: str) -> None:
        # Repeated relation keys represent a state update.  Replace that key
        # instead of allowing an older location to win by chance.
        for index, (old_key, _, _) in enumerate(self.entries):
            if float(torch.dot(old_key, key)) > 0.99999:
                self.entries[index] = (key.clone(), value.clone(), label)
                return
        self.entries.append((key.clone(), value.clone(), label))
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def read(self, key: torch.Tensor) -> str | None:
        if not self.entries:
            return None
        scores = [float(torch.dot(old_key, key)) for old_key, _, _ in self.entries]
        return self.entries[max(range(len(scores)), key=lambda index: scores[index])][2]


def _copy_event(event: dict[str, Any]) -> dict[str, Any]:
    return dict(event)


def _map_location(location: str, target_scene: str) -> str:
    suffix = str(location).split("/", 1)[-1]
    mapping = {
        "kitchen table": "kitchen table", "counter": "counter", "shelf": "shelf",
        "living room shelf": "shelf", "desk": "desk", "side table": "table",
    }
    return f"{target_scene}/{mapping.get(suffix, suffix)}"


def condition_stream(episode: dict[str, Any], condition: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return events and transformed gold answers for one experimental arm."""

    events = [_copy_event(event) for event in episode["events"]]
    moved = episode["moved_object"]
    query_gold = {str(query["id"]): str(query["answer"]) for query in episode["queries"]}
    if condition == "relocated":
        return events, query_gold
    if condition == "same_scene_control":
        transformed: list[dict[str, Any]] = []
        for event in events:
            if event["kind"] == "scene_change":
                transformed.append({
                    "kind": "scene_continue", "scene": "A", "time": int(event["time"]),
                    "from_scene": "A", "to_scene": "A",
                })
                continue
            event["scene"] = "A"
            for field in ("location", "from_location", "to_location"):
                if field in event:
                    event[field] = _map_location(str(event[field]), "A")
            if event.get("kind") == "forecast" and "action" in event:
                event["action"] = str(event["action"]).replace("living room shelf", "shelf").replace("side table", "table")
            transformed.append(event)
        answers = dict(query_gold)
        for query in episode["queries"]:
            if query["type"] == "location":
                answers[str(query["id"])] = _map_location(query["answer"], "A")
            elif query["type"] == "next_action":
                answers[str(query["id"])] = query["answer"].replace("living room shelf", "shelf").replace("side table", "table")
        return transformed, answers
    if condition == "filler_only_control":
        # Match the event-card and key/value write budgets while removing the
        # target move.  The scene marker becomes a same-scene marker and the
        # four move bindings become four unrelated filler bindings.
        transformed = []
        for original in events:
            if original["kind"] == "forecast":
                continue
            if original["kind"] == "scene_change":
                transformed.append({
                    "kind": "scene_continue", "scene": "A", "time": int(original["time"]),
                    "from_scene": "A", "to_scene": "A",
                })
                continue
            if original["kind"] == "move":
                transformed.append({
                    "kind": "filler", "scene": "A", "time": int(original["time"]),
                    "object": "control distractor", "location": "A/counter",
                    "state": "intact", "actor": "control", "control_records": 4,
                })
                continue
            event = _copy_event(original)
            event["scene"] = "A"
            for field in ("location", "from_location", "to_location"):
                if field in event:
                    event[field] = _map_location(str(event[field]), "A")
            transformed.append(event)
        answers = dict(query_gold)
        initial = next(event for event in transformed
                       if event.get("kind") == "place" and event.get("object") == moved)
        for query in episode["queries"]:
            qid = str(query["id"])
            if query["type"] == "location":
                answers[qid] = initial["location"]
            elif query["type"] == "state":
                answers[qid] = "intact"
            elif query["type"] == "mover":
                answers[qid] = "unknown"
            elif query["type"] == "next_action":
                answers[qid] = "unknown"
        return transformed, answers
    raise ValueError(f"unknown condition: {condition}")


def event_records(event: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Convert one structured event into (relation, object, label, scene)."""

    kind = str(event.get("kind"))
    scene = str(event.get("scene", "A"))
    obj = str(event.get("object", ""))
    if kind == "place":
        return [("location", obj, str(event["location"]), scene),
                ("state", obj, str(event.get("state", "intact")), scene),
                ("mover", obj, "unknown", scene),
                ("moved_status", obj, "not_moved", scene)]
    if kind == "owner":
        return [("owner", obj, str(event["owner"]), scene)]
    if kind == "move":
        return [("location", obj, str(event["to_location"]), scene),
                ("mover", obj, str(event["actor"]), scene),
                ("state", obj, str(event["state_after"]), scene),
                ("moved_status", obj, "moved", scene)]
    if kind == "forecast":
        # The forecast is keyed by its actor; the query targets the moved object
        # and therefore receives a separate synthetic key below.
        return []
    if kind == "filler":
        count = max(1, int(event.get("control_records", 1)))
        return [("filler", obj + "/" + str(event["time"]) + f"/{index}",
                 str(event.get("state", "intact")), scene) for index in range(count)]
    return []


def query_key_and_label(query: dict[str, Any], answer: str, scene: str,
                        codebook: SemanticCodebook, drift: float) -> tuple[torch.Tensor, torch.Tensor]:
    relation = str(query["type"])
    obj = str(query["object"])
    return codebook.key(relation, obj, scene, drift), codebook.value(relation, answer)


def write_event(memory: Any, event: dict[str, Any], *, codebook: SemanticCodebook,
                drift: float) -> None:
    records = event_records(event)
    if not records:
        return
    if isinstance(memory, LaCTMemory):
        keys, values = [], []
        for relation, obj, label, scene in records:
            keys.append(codebook.key(relation, obj, scene, drift))
            values.append(codebook.value(relation, label))
        memory.write(torch.stack(keys), torch.stack(values))
    else:
        for relation, obj, label, scene in records:
            memory.write(codebook.key(relation, obj, scene, drift),
                         codebook.value(relation, label), label)


def decode_lact(memory: LaCTMemory, query: dict[str, Any], candidates: list[str], *, scene: str,
                codebook: SemanticCodebook, drift: float) -> str | None:
    key = codebook.key(str(query["type"]), str(query["object"]), scene, drift)
    output = memory.read(key[None])[0]
    values = torch.stack([codebook.value(str(query["type"]), candidate) for candidate in candidates])
    scores = F.normalize(output[None], dim=-1) @ F.normalize(values, dim=-1).T
    return candidates[int(scores.argmax().item())]


def decode_table(memory: ExplicitRetrieval, query: dict[str, Any], candidates: list[str], *, scene: str,
                 codebook: SemanticCodebook, drift: float) -> str | None:
    return memory.read(codebook.key(str(query["type"]), str(query["object"]), scene, drift))


def _relation_query(query: dict[str, Any], relation: str) -> dict[str, Any]:
    return {"type": relation, "object": str(query["object"])}


def predict_next_action(memory: Any, query: dict[str, Any], *, arm: str, scene: str,
                        codebook: SemanticCodebook, drift: float,
                        candidates: dict[str, list[str]]) -> str | None:
    """Compose the held-out future action from retained current state."""
    decoder = decode_lact if arm == "lact" else decode_table
    moved = decoder(memory, _relation_query(query, "moved_status"),
                    candidates["moved_status"], scene=scene,
                    codebook=codebook, drift=drift)
    if moved != "moved":
        return "unknown"
    owner = decoder(memory, _relation_query(query, "owner"), candidates["owner"],
                    scene=scene, codebook=codebook, drift=drift)
    location = decoder(memory, _relation_query(query, "location"), candidates["location"],
                       scene=scene, codebook=codebook, drift=drift)
    if owner is None or location is None:
        return None
    return f"{owner} checks the {str(location).split('/', 1)[-1]}"


def _candidate_sets(episodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    result = {name: [] for name in (*QUERY_TYPES, "moved_status")}
    for episode in episodes:
        for condition in CONDITIONS:
            events, _ = condition_stream(episode, condition)
            for event in events:
                for relation, _, value, _ in event_records(event):
                    if relation in result and value not in result[relation]:
                        result[relation].append(value)
    result["moved_status"] = ["not_moved", "moved"]
    for name in QUERY_TYPES:
        for value in ("unknown",):
            if value not in result[name]:
                result[name].append(value)
    return result


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(args.data.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {"experiment_c_relocation_v1", "experiment_c_relocation_v2"}:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    episodes = manifest.get("episodes", [])
    if args.limit:
        episodes = episodes[:args.limit]
    codebook = SemanticCodebook(args.dim, int(manifest.get("seed", 0)) + args.seed)
    candidates = _candidate_sets(episodes)
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for episode in episodes:
            events, answers = condition_stream(episode, condition)
            # The held-out future card must never be written to memory.
            events = [event for event in events if event.get("kind") != "forecast"]
            for trial in range(args.trials):
                trial_seed = args.seed + trial * 1009 + int(episode["seed"])
                torch.manual_seed(trial_seed)
                memories: dict[str, Any] = {
                    "oracle": None,
                    "explicit_retrieval": ExplicitRetrieval(args.max_entries),
                    "lact": LaCTMemory(args.dim, head_dim=args.head_dim, base_lr=args.base_lr,
                                       use_muon=not args.no_muon, projection="identity",
                                       seed=trial_seed),
                    "none": None,
                }
                for event in events:
                    for name in ("explicit_retrieval", "lact"):
                        write_event(memories[name], event, codebook=codebook,
                                    drift=args.scene_drift_strength)
                query_scene = "B" if condition == "relocated" else "A"
                for query in episode["queries"]:
                    qid, qtype = str(query["id"]), str(query["type"])
                    gold = str(answers[qid])
                    for arm in ARMS:
                        if arm == "oracle":
                            prediction = gold
                        elif arm == "none":
                            prediction = None
                        elif arm == "explicit_retrieval":
                            if qtype == "next_action":
                                prediction = predict_next_action(
                                    memories[arm], query, arm=arm, scene=query_scene,
                                    codebook=codebook, drift=args.scene_drift_strength,
                                    candidates=candidates,
                                )
                            else:
                                prediction = decode_table(memories[arm], query, candidates[qtype],
                                                          scene=query_scene, codebook=codebook,
                                                          drift=args.scene_drift_strength)
                        else:
                            if qtype == "next_action":
                                prediction = predict_next_action(
                                    memories[arm], query, arm=arm, scene=query_scene,
                                    codebook=codebook, drift=args.scene_drift_strength,
                                    candidates=candidates,
                                )
                            else:
                                prediction = decode_lact(memories[arm], query, candidates[qtype],
                                                         scene=query_scene, codebook=codebook,
                                                         drift=args.scene_drift_strength)
                        rows.append({
                            "episode_id": episode["episode_id"], "condition": condition,
                            "trial": trial, "query_id": qid, "query_type": qtype,
                            "question": query["question"], "gold": gold,
                            "prediction": prediction, "correct": prediction == gold,
                            "arm": arm, "scene_changed": condition == "relocated",
                            "filler_before": manifest["protocol"]["filler_before"],
                            "filler_after": manifest["protocol"]["filler_after"],
                            "scene_drift_strength": args.scene_drift_strength,
                        })
    summary: dict[str, Any] = {
        "schema": "experiment_c_result_v2", "data": str(args.data),
        "episodes": len(episodes), "trials": args.trials, "dim": args.dim,
        "head_dim": args.head_dim, "base_lr": args.base_lr,
        "scene_drift_strength": args.scene_drift_strength,
        "forecast_event_withheld": True,
        "next_action_policy": "after a move, the owner checks the object's final location",
        "by_arm_condition": {}, "by_arm_type": {},
    }
    for group_name, key_fields in (("by_arm_condition", ("arm", "condition")),
                                   ("by_arm_type", ("arm", "query_type"))):
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in key_fields)].append(row)
        for key, group in sorted(groups.items()):
            valid = [row for row in group if row["prediction"] is not None]
            summary[group_name]["::".join(key)] = {
                "n": len(group), "coverage": round(len(valid) / len(group), 4),
                "accuracy": round(sum(row["correct"] for row in valid) / len(valid), 4) if valid else None,
            }
    summary["paired_relocated_minus_same_scene"] = {}
    for arm in ARMS:
        for qtype in QUERY_TYPES:
            def acc(condition: str) -> float | None:
                group = [row for row in rows if row["arm"] == arm and row["query_type"] == qtype
                         and row["condition"] == condition and row["prediction"] is not None]
                return sum(row["correct"] for row in group) / len(group) if group else None
            left, right = acc("relocated"), acc("same_scene_control")
            summary["paired_relocated_minus_same_scene"][f"{arm}::{qtype}"] = (
                None if left is None or right is None else round(left - right, 4)
            )
    return rows, summary


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment C: controlled relocation and rearrangement",
        "",
        "> This is a synthetic mechanism probe. It is not a result on real household video.",
        "",
        "The observed stream writes object–location, object–owner, mover and state events. "
        "The held-out future event is never written. `next_action` is composed from retained "
        "moved-status, owner and final-location bindings under a fixed public policy. "
        "`relocated`, `same_scene_control` and `filler_only_control` each contain 16 event cards "
        "and 27 key/value writes. The relocated/same-scene contrast also changes two "
        "room-specific location strings, so it is not a pure scene-cue intervention. "
        "`explicit_retrieval` is a fixed-size table control, and `oracle` is a ceiling.",
        "",
        f"- Episodes: **{summary['episodes']}**; trials: **{summary['trials']}**",
        f"- Semantic code dimension: **{summary['dim']}**; scene drift strength: **{summary['scene_drift_strength']}**",
        f"- Rows: **{len(rows)}**; no external video or model download was used",
        "",
        "## Accuracy by arm and condition",
        "",
        "| arm | condition | n | coverage | accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for key, metric in summary["by_arm_condition"].items():
        arm, condition = key.split("::")
        lines.append(f"| {arm} | {condition} | {metric['n']} | {metric['coverage']:.3f} | "
                     f"{metric['accuracy'] if metric['accuracy'] is not None else '—'} |")
    lines += ["", "## Accuracy by query type", "",
              "| arm | query type | n | coverage | accuracy |", "|---|---|---:|---:|---:|"]
    for key, metric in summary["by_arm_type"].items():
        arm, query_type = key.split("::")
        lines.append(f"| {arm} | {query_type} | {metric['n']} | {metric['coverage']:.3f} | "
                     f"{metric['accuracy'] if metric['accuracy'] is not None else '—'} |")
    lines += ["", "## Relocated minus same-scene accuracy", "",
              "A negative value means the full relocated condition hurt this query type. "
              "Because two location strings also change, this is not a pure scene-cue effect "
              "and remains a diagnostic rather than a claim about real video.", "",
              "| arm | query type | delta |", "|---|---|---:|"]
    for key, delta in summary["paired_relocated_minus_same_scene"].items():
        arm, qtype = key.split("::")
        lines.append(f"| {arm} | {qtype} | {delta if delta is not None else '—'} |")
    lines += ["", "## Interpretation boundary", "",
              "The codebook removes OCR and visual encoder confounds so this run answers only "
              "whether a fixed-size associative state can retain bindings through filler and a "
              "relocation-condition change. A VLM follow-up should render the same event manifest into a "
              "video and repeat the conditions with equal frame budgets. The next-action score "
              "tests state-conditioned composition under the stated policy, not open-world action "
              "forecasting. `none` has zero coverage by design; it is a blind control, not a "
              "failed model run.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/experiment_c/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_c"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=0, help="deprecated alias for --limit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=0.05)
    parser.add_argument("--max-entries", type=int, default=128)
    parser.add_argument("--scene-drift-strength", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-muon", action="store_true")
    args = parser.parse_args()
    args.limit = args.limit or args.episodes
    if args.trials <= 0 or args.dim <= 0 or args.head_dim <= 0 or args.dim % args.head_dim:
        raise SystemExit("trials/dim/head-dim are invalid; dim must be divisible by head-dim")
    rows, summary = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = args.report or args.output.with_suffix(".md")
    write_report(report, summary, rows)
    print(json.dumps({"rows": len(rows), "summary": str(args.output.with_suffix('.summary.json')),
                      "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
