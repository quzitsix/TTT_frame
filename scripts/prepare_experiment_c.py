#!/usr/bin/env python3
"""Create the small, dependency-free relocation control set used by experiment C.

The benchmark is deliberately a *mechanism* fixture.  It has the same event
schema as a future VirtualHome/Habitat export, but does not require either
dataset or a simulator.  Each episode has three objects and three residents:
the objects are assigned owners, one object is moved by another resident, and
its state changes.  Unrelated filler events are inserted before and after the
move.  The generated PNG cards make the sequence easy to inspect manually;
the evaluator uses the structured events and a deterministic semantic codebook
so a failure can be attributed to the memory rather than OCR or video decoding.

Examples
--------
    python scripts/prepare_experiment_c.py --output-dir data/experiment_c
    python scripts/evaluate_experiment_c.py --data data/experiment_c/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont


OBJECTS = ("red mug", "blue bowl", "green cup")
PEOPLE = ("Alice", "Bob", "Carol")
LOCATIONS_A = ("A/kitchen table", "A/counter", "A/shelf")
LOCATIONS_B = ("B/living room shelf", "B/desk", "B/side table")
STATES = ("intact", "open", "used")
FILLER_OBJECTS = ("yellow plate", "white towel", "black pan", "wooden tray")


def _episode_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"experiment-c/{seed}/{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _event(kind: str, *, scene: str, time: int, **fields) -> dict:
    row = {"time": time, "kind": kind, "scene": scene}
    row.update(fields)
    return row


def build_episode(index: int, *, seed: int, filler_before: int, filler_after: int) -> dict:
    rng = random.Random(_episode_seed(seed, index))
    objects = list(OBJECTS)
    owners = list(PEOPLE)
    rng.shuffle(owners)
    owner_of = dict(zip(objects, owners))
    initial_location = dict(zip(objects, LOCATIONS_A))
    # Rotate rather than sample so every object is moved equally often across
    # the default twelve episodes.
    moved_object = objects[index % len(objects)]
    # Always choose one of the two non-owners.  The old ``+ 1 + index`` rule
    # wrapped back to the owner for every third episode, coupling the mover and
    # owner questions and making that subset artificially easy.
    mover = PEOPLE[(PEOPLE.index(owner_of[moved_object]) + 1 + (index % 2)) % len(PEOPLE)]
    final_location = LOCATIONS_B[index % len(LOCATIONS_B)]
    final_state = STATES[(index + 1) % len(STATES)]

    events: list[dict] = []
    t = 0
    for obj in objects:
        events.append(_event(
            "place", scene="A", time=t, object=obj,
            location=initial_location[obj], state="intact", actor=owner_of[obj],
        ))
        t += 1
        events.append(_event(
            "owner", scene="A", time=t, object=obj, owner=owner_of[obj],
        ))
        t += 1
    for filler_index in range(filler_before):
        obj = FILLER_OBJECTS[filler_index % len(FILLER_OBJECTS)]
        events.append(_event(
            "filler", scene="A", time=t, object=obj,
            location=LOCATIONS_A[(filler_index + index) % len(LOCATIONS_A)],
            state="intact", actor=PEOPLE[filler_index % len(PEOPLE)],
        ))
        t += 1

    # This marker is intentionally explicit.  The evaluator can therefore
    # report scene-change effects separately from ordinary filler writes.
    events.append(_event("scene_change", scene="B", time=t, from_scene="A", to_scene="B"))
    t += 1
    events.append(_event(
        "move", scene="B", time=t, object=moved_object, actor=mover,
        from_location=initial_location[moved_object], to_location=final_location,
        state_before="intact", state_after=final_state,
    ))
    t += 1
    for filler_index in range(filler_after):
        obj = FILLER_OBJECTS[(filler_index + 1) % len(FILLER_OBJECTS)]
        events.append(_event(
            "filler", scene="B", time=t, object=obj,
            location=LOCATIONS_B[(filler_index + index + 1) % len(LOCATIONS_B)],
            state="intact", actor=PEOPLE[(filler_index + 1) % len(PEOPLE)],
        ))
        t += 1

    next_action = f"{owner_of[moved_object]} checks the {final_location.split('/', 1)[1]}"
    future_event = _event(
        "forecast", scene="B", time=t, actor=owner_of[moved_object],
        object=moved_object, action=next_action,
    )

    queries = [
        {"id": f"{index:03d}-location", "type": "location", "object": moved_object,
         "question": f"Where is the {moved_object} after the move?", "answer": final_location},
        {"id": f"{index:03d}-mover", "type": "mover", "object": moved_object,
         "question": f"Who moved the {moved_object}?", "answer": mover},
        {"id": f"{index:03d}-owner", "type": "owner", "object": moved_object,
         "question": f"Who does the {moved_object} belong to?", "answer": owner_of[moved_object]},
        {"id": f"{index:03d}-state", "type": "state", "object": moved_object,
         "question": f"What is the state of the {moved_object} after the move?", "answer": final_state},
        {"id": f"{index:03d}-forecast", "type": "next_action", "object": moved_object,
         "question": f"What will {owner_of[moved_object]} most likely do next?", "answer": next_action},
    ]
    return {
        "episode_id": f"c-{index:03d}", "seed": _episode_seed(seed, index),
        "objects": objects, "people": list(PEOPLE), "moved_object": moved_object,
        "mover": mover, "events": events, "future_event": future_event,
        "queries": queries,
        "conditions": {
            "relocated": {"write_scene": "A", "query_scene": "B", "scene_changed": True},
            "same_scene_control": {"write_scene": "A", "query_scene": "A", "scene_changed": False},
            "filler_only_control": {"write_scene": "A", "query_scene": "A", "scene_changed": False},
        },
    }


def _draw_card(event: dict, path: Path, *, index: int, total: int) -> None:
    scene = event.get("scene", "A")
    bg = (236, 242, 250) if scene == "A" else (250, 238, 232)
    image = Image.new("RGB", (1280, 720), bg)
    draw = ImageDraw.Draw(image)
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        header_font = ImageFont.truetype(font_path, 30)
        body_font = ImageFont.truetype(font_path, 42)
    except OSError:
        header_font = body_font = ImageFont.load_default()
    draw.rectangle((0, 0, 1280, 90), fill=(50, 75, 105) if scene == "A" else (110, 70, 50))
    draw.text((35, 25), f"Experiment C · {scene} · event {index + 1}/{total}", fill="white", font=header_font)
    lines = [f"kind: {event['kind']}", f"time: {event['time']}"]
    for key, value in event.items():
        if key not in {"kind", "time", "scene"}:
            lines.append(f"{key}: {value}")
    y = 150
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=65):
            draw.text((60, y), wrapped, fill=(20, 25, 35), font=body_font)
            y += 62
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiment_c"))
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--filler-before", type=int, default=4)
    parser.add_argument("--filler-after", type=int, default=4)
    args = parser.parse_args()
    if args.episodes <= 0 or args.filler_before < 0 or args.filler_after < 0:
        raise SystemExit("episodes must be positive and filler counts nonnegative")
    root = args.output_dir.resolve()
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    # This directory is generator-owned.  Clear every prior episode card so a
    # later run with fewer episodes cannot leave misleading stale files.
    for stale in frames.glob("c-*.png"):
        stale.unlink()
    episodes = []
    for index in range(args.episodes):
        episode = build_episode(index, seed=args.seed,
                                filler_before=args.filler_before,
                                filler_after=args.filler_after)
        episode["frame_paths"] = []
        for frame_index, event in enumerate(episode["events"]):
            path = frames / f"{episode['episode_id']}_{frame_index:03d}.png"
            _draw_card(event, path, index=frame_index, total=len(episode["events"]))
            episode["frame_paths"].append(str(path.relative_to(root)))
        episodes.append(episode)
    manifest = {
        "schema": "experiment_c_relocation_v2",
        "seed": args.seed, "episodes": episodes,
        "protocol": {
            "filler_before": args.filler_before, "filler_after": args.filler_after,
            "conditions": ["relocated", "same_scene_control", "filler_only_control"],
            "query_types": ["location", "mover", "owner", "state", "next_action"],
            "future_target_withheld": True,
            "note": "Observed PNG cards exclude future_event; the symbolic evaluator also withholds it.",
        },
    }
    destination = root / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(destination), "episodes": len(episodes),
                      "queries": sum(len(item["queries"]) for item in episodes),
                      "frames": sum(len(item["events"]) for item in episodes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
