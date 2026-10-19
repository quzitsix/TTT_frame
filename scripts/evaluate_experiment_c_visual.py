#!/usr/bin/env python3
"""Evaluate the visual PNG-card proxy for controlled experiment C.

The runner materializes relocated, same-scene and filler-only streams with
matched frame budgets, always withholds the future target, and compares native
Qwen3-VL ICL, text-only blind, Spatial-TTT episode memory, a trained
Spatial-read-path slow-only ablation, and a base-attention bypass.  The default
strict MCQ path asks one query per generation and balances correct A/B/C/D
positions.  A batched layout remains available for diagnosis.  Open answers
remain a diagnostic only.  These text-heavy synthetic cards test plumbing and
readout; they are not a substitute for real first-person video.

Examples::

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_experiment_c_visual.py \
      --data data/experiment_c/manifest.json --episodes 4 --arm icl \
      --device cuda:2

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_experiment_c_visual.py \
      --data data/experiment_c/manifest.json --episodes 4 --arm spatial \
      --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
      --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
      --device cuda:2
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import string
import sys
import textwrap
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory  # noqa: E402
from scripts.evaluate_experiment_c import condition_stream  # noqa: E402


CONDITIONS = ("relocated", "same_scene_control", "filler_only_control")
SPATIAL_ARMS = ("spatial", "spatial_slow_only", "spatial_base")
EVENT_CARD_INSTRUCTION = (
    "Frames are chronological. Read only the visible event cards. Answer briefly."
)


def _config_fingerprint(model: Any) -> str:
    config = model.config.to_dict()
    for key in ("_name_or_path", "transformers_version", "_commit_hash"):
        config.pop(key, None)
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _tokens(value: str) -> list[str]:
    value = str(value or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"[a-z0-9]+", value)


def _canonical_text(value: str, query_type: str) -> str:
    value = str(value or "").strip()
    if query_type in {"location", "next_action"}:
        value = re.sub(r"\b[AB]/", "", value, flags=re.IGNORECASE)
    return value


def _semantic_correct(prediction: str, gold: str, query_type: str) -> bool:
    predicted = _tokens(_canonical_text(prediction, query_type))
    target = _tokens(_canonical_text(gold, query_type))
    if not target:
        return False
    if target == ["unknown"]:
        joined = " ".join(predicted)
        if re.search(r"\b(?:not|isnt|isn)\b(?:\s+\w+){0,2}\s+unknown\b", joined):
            return False
        return any(token in predicted for token in ("unknown", "none", "nobody"))
    joined = " ".join(predicted)
    target_text = " ".join(target)
    if re.search(rf"\b(?:not|isnt|isn)\b(?:\s+\w+){{0,2}}\s+{re.escape(target_text)}\b", joined):
        return False
    if query_type in {"mover", "owner", "state"}:
        conflicts = ({"alice", "bob", "carol"} if query_type in {"mover", "owner"}
                     else {"intact", "open", "used"}) - set(target)
        return (len(target) == 1 and target[0] in predicted
                and not any(value in predicted for value in conflicts))
    if query_type == "location":
        return all(token in predicted for token in target)
    if query_type == "next_action":
        # Require the person, final location and the specified check action.
        # This avoids giving credit to generic future-action prose.
        required = [token for token in target if token not in {"the", "checks"}]
        has_check = any(token.startswith("check") for token in predicted)
        people = {"alice", "bob", "carol"} - set(target)
        return (has_check and all(token in predicted for token in required)
                and "not" not in predicted and not any(value in predicted for value in people))
    return target == predicted


def score_answer(prediction: str, gold: str, query_type: str) -> dict[str, Any]:
    predicted = _tokens(_canonical_text(prediction, query_type))
    target = _tokens(_canonical_text(gold, query_type))
    overlap = sum((Counter(predicted) & Counter(target)).values())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    ptext, gtext = " ".join(predicted), " ".join(target)
    return {"exact_normalized": ptext == gtext,
            "gold_substring": bool(gtext) and gtext in ptext,
            "token_f1": round(f1, 4),
            "semantic_correct": _semantic_correct(prediction, gold, query_type)}


def _load_images(manifest_path: Path, episode: dict[str, Any], max_side: int) -> list[Any]:
    from PIL import Image

    root = manifest_path.parent
    result = []
    for relative in episode["frame_paths"]:
        path = root / relative
        image = Image.open(path).convert("RGB")
        image.thumbnail((max_side, max_side))
        result.append(image.copy())
    return result


def _render_events(events: list[dict[str, Any]], max_side: int) -> list[Any]:
    """Render a condition stream in memory using the same card vocabulary."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        header_font = ImageFont.truetype(font_path, 30)
        body_font = ImageFont.truetype(font_path, 42)
    except OSError:
        header_font = body_font = ImageFont.load_default()
    result = []
    total = len(events)
    for index, event in enumerate(events):
        scene = str(event.get("scene", "A"))
        bg = (236, 242, 250) if scene == "A" else (250, 238, 232)
        image = Image.new("RGB", (1280, 720), bg)
        draw = ImageDraw.Draw(image)
        header = (50, 75, 105) if scene == "A" else (110, 70, 50)
        draw.rectangle((0, 0, 1280, 90), fill=header)
        draw.text((35, 25), f"Experiment C · {scene} · event {index + 1}/{total}",
                  fill="white", font=header_font)
        lines = [f"kind: {event['kind']}", f"time: {event['time']}"]
        lines.extend(f"{key}: {value}" for key, value in event.items()
                     if key not in {"kind", "time", "scene"})
        y = 150
        for line in lines:
            for wrapped in textwrap.wrap(str(line), width=65):
                draw.text((60, y), wrapped, fill=(20, 25, 35), font=body_font)
                y += 62
        image.thumbnail((max_side, max_side))
        result.append(image)
    return result


def _uniform(images: list[Any], limit: int) -> list[Any]:
    if not limit or len(images) <= limit:
        return images
    if limit == 1:
        return [images[len(images) // 2]]
    indices = [round(index * (len(images) - 1) / (limit - 1)) for index in range(limit)]
    return [images[index] for index in indices]


class NativeICL:
    def __init__(self, model_path: str, device: str, dtype: str, local_files_only: bool):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=local_files_only, dtype=self.dtype,
            device_map={"": str(self.device)}, attn_implementation="sdpa",
        ).eval()
        self.base_fingerprint = _config_fingerprint(self.model)

    def _move(self, batch):
        return {key: value.to(self.device, dtype=self.dtype if value.is_floating_point() else value.dtype)
                if torch.is_tensor(value) else value for key, value in batch.items()}

    @torch.inference_mode()
    def answer(self, images: list[Any], question: str, max_new_tokens: int) -> str:
        import numpy as np
        from transformers.video_utils import VideoMetadata

        frames = list(images)
        while len(frames) < 3:
            frames.append(frames[-1].copy())
        array = np.stack([np.asarray(image) for image in frames])
        metadata = VideoMetadata(total_num_frames=len(frames), fps=1.0,
                                 frames_indices=list(range(len(frames))))
        prompt = self.processor.apply_chat_template([{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text":
             EVENT_CARD_INSTRUCTION + "\n" + question},
        ]}], tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], videos=[array], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt")
        batch = self._move(batch)
        output = self.model.generate(**batch, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
        continuation = output[:, batch["input_ids"].shape[1]:]
        return self.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def answer_text(self, question: str, max_new_tokens: int) -> str:
        prompt = self.processor.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": EVENT_CARD_INSTRUCTION + "\n" + question},
        ]}], tokenize=False, add_generation_prompt=True)
        batch = self._move(self.processor(text=[prompt], return_tensors="pt"))
        output = self.model.generate(**batch, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
        continuation = output[:, batch["input_ids"].shape[1]:]
        return self.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()


def _query_prompt(query: dict[str, Any], *, forecast_hidden: bool,
                  condition: str, answer_mode: str) -> str:
    query_type = str(query["type"])
    if query_type == "next_action" and forecast_hidden:
        question = (
            "Policy: if an object was moved, its owner next checks its final location; "
            "if no move occurred, the result is unknown. What happens next?"
        )
        return (question if answer_mode == "mcq" else question +
                " Answer exactly as '<owner> checks the <location>' or 'unknown'.")
    question = str(query["question"])
    if condition == "filler_only_control":
        obj = str(query["object"])
        question = {
            "location": f"Where is the {obj} at the end of the sequence?",
            "mover": f"Who moved the {obj}?",
            "owner": f"Who does the {obj} belong to?",
            "state": f"What is the state of the {obj} at the end of the sequence?",
        }.get(query_type, question)
    if answer_mode == "mcq":
        return question
    suffix = {
        "location": " Answer only the location; omit any A/ or B/ scene prefix.",
        "mover": " Answer only the person's name, or unknown.",
        "owner": " Answer only the person's name.",
        "state": " Answer only the state.",
        "next_action": " Answer with only the predicted action.",
    }.get(query_type, " Answer briefly.")
    return question + suffix


def _display_answer(value: str, query_type: str) -> str:
    return _canonical_text(str(value), query_type).strip()


def _answer_pools(manifest: dict[str, Any], conditions: list[str]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    for episode in manifest.get("episodes", []):
        for condition in conditions:
            _, answers = condition_stream(episode, condition)
            for query in episode["queries"]:
                qtype = str(query["type"])
                value = _display_answer(answers[str(query["id"])], qtype)
                if value not in pools[qtype]:
                    pools[qtype].append(value)
    for qtype in ("location", "mover", "owner", "state", "next_action"):
        if "unknown" not in pools[qtype]:
            pools[qtype].append("unknown")
    return dict(pools)


def _choices(query: dict[str, Any], gold: str, *, episode_id: str,
             condition: str, pools: dict[str, list[str]]) -> tuple[list[str], str]:
    qtype = str(query["type"])
    answer = _display_answer(gold, qtype)
    distractors = [value for value in pools[qtype] if value != answer]
    seed_text = f"experiment-c-v2/{episode_id}/{condition}/{query['id']}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    rng.shuffle(distractors)
    options = distractors[:3]
    if len(options) < 3:
        raise ValueError(f"not enough distinct {qtype} choices")
    episode_index = int(str(episode_id).rsplit("-", 1)[-1])
    condition_index = CONDITIONS.index(condition)
    type_index = ("location", "mover", "owner", "state", "next_action").index(qtype)
    answer_index = (episode_index + condition_index + type_index) % 4
    options.insert(answer_index, answer)
    return options, "ABCD"[answer_index]


def _mcq_prompt(specs: list[dict[str, Any]]) -> str:
    if len(specs) == 1:
        lines = ["The event cards are chronological. Answer the multiple-choice question.",
                 "Return only one capital letter: A, B, C, or D."]
    else:
        lines = [
            "The event cards are chronological. Answer all multiple-choice questions.",
            f"Return only {len(specs)} capital letters in question order, for example: A B C D A.",
        ]
    for index, spec in enumerate(specs, 1):
        lines.extend(["", f"{index}. {spec['prompt']}"])
        lines.extend(f"{letter}. {value}" for letter, value in zip("ABCD", spec["options"]))
    return "\n".join(lines)


def _parse_mcq_letters(response: str, expected: int) -> list[str]:
    text = str(response or "").strip().upper()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if (isinstance(parsed, list) and len(parsed) == expected
            and all(item in ("A", "B", "C", "D") for item in parsed)):
        return parsed
    # Prefer a clean compact/list-shaped response.  We intentionally do not
    # scan arbitrary prose because words beginning with A-D cause false hits.
    repeated = max(0, expected - 1)
    compact = re.fullmatch(
        rf"[\[\](){{}}\s,;:'\".-]*([A-D](?:[\s,;]+[A-D]){{{repeated}}})"
        rf"[\[\](){{}}\s,;:'\".-]*", text,
    )
    if compact:
        letters = re.findall(r"[A-D]", compact.group(1))
        if len(letters) == expected:
            return letters
    if re.fullmatch(r"[A-D]{%d}" % expected, text):
        return list(text)
    numbered = re.findall(r"(?:^|\n)\s*\d+[.)]\s*([A-D])(?:\b|[.)])", text)
    return numbered if len(numbered) == expected else []


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.data.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {"experiment_c_relocation_v1", "experiment_c_relocation_v2"}:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    start = args.episode_offset
    stop = None if not args.episodes else start + args.episodes
    episodes = manifest.get("episodes", [])[start:stop]
    if not episodes:
        raise ValueError("manifest contains no episodes")
    pools = _answer_pools(manifest, list(CONDITIONS))
    # Loading one large VLM twice can exhaust a 24 GB card.  Run arms one at a
    # time and let the caller invoke this command again for the other arm.
    icl = None
    spatial = None
    if args.arm in {"icl", "blind"}:
        icl = NativeICL(args.model_path, args.device, args.dtype, args.local_files_only)
    elif args.arm in SPATIAL_ARMS:
        config = SpatialVideoConfig(model_path=args.model_path, device=args.device,
                                    dtype=args.dtype, local_files_only=args.local_files_only,
                                    spatial_checkpoint=args.spatial_checkpoint,
                                    chunk_seconds=3600.0, frames_per_chunk=4096,
                                    max_side=args.max_side, max_new_tokens=args.max_new_tokens)
        spatial = SpatialVideoMemory(config)

    base_fingerprint = icl.base_fingerprint if icl is not None else spatial.base_fingerprint
    checkpoint_hash = spatial.spatial_checkpoint_hash if spatial is not None else None
    pair_family = "spatial_official" if spatial is not None else "qwen_base"
    comparison_payload = {
        "manifest_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "manifest_schema": manifest.get("schema"),
        "model_path": str(Path(args.model_path).resolve()),
        "base_fingerprint": base_fingerprint,
        "spatial_checkpoint_hash": checkpoint_hash,
        "dtype": args.dtype,
        "max_frames": args.max_frames,
        "max_side": args.max_side,
        "max_new_tokens": args.max_new_tokens,
        "answer_mode": args.answer_mode,
        "mcq_layout": args.mcq_layout if args.answer_mode == "mcq" else None,
        "conditions": list(args.conditions),
        "pair_family": pair_family,
    }
    run_protocol = {
        **comparison_payload,
        "comparison_fingerprint": hashlib.sha256(
            json.dumps(comparison_payload, sort_keys=True).encode()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "spatial_checkpoint": (
            str(Path(args.spatial_checkpoint).resolve()) if spatial is not None else None
        ),
    }

    def invoke(images: list[Any], prompt: str) -> str:
        if args.arm == "icl":
            return icl.answer(images, prompt, args.max_new_tokens)
        if args.arm == "blind":
            return icl.answer_text(prompt, args.max_new_tokens)
        return spatial.answer(
            prompt, use_memory=args.arm in {"spatial", "spatial_slow_only"},
            max_new_tokens=args.max_new_tokens,
        )

    rows = []
    for episode in episodes:
        for condition in args.conditions:
            events, gold_by_id = condition_stream(episode, condition)
            # Never expose a future target, including when reading a legacy v1
            # manifest that stored it in the observed event list.
            events = [event for event in events if event.get("kind") != "forecast"]
            images = _uniform(_render_events(events, args.max_side), args.max_frames)
            if not images:
                raise ValueError(f"episode {episode['episode_id']} condition {condition} has no frame cards")
            if spatial is not None:
                spatial.reset()
                spatial.ingest_frames(images, [float(index) for index in range(len(images))])
                spatial.finish_ingest()
                if args.arm == "spatial_slow_only":
                    spatial.discard_fast_state()
            encode_passes = (
                len(episode["queries"])
                if args.arm == "icl" and (args.answer_mode == "open" or args.mcq_layout == "separate")
                else 1 if args.arm in {"icl", *SPATIAL_ARMS}
                else 0
            )
            episode_state = {
                "spatial": "episode_fast_state",
                "spatial_slow_only": "learned_initial_state",
                "spatial_base": "spatial_branch_bypassed",
                "icl": "visual_context",
                "blind": "no_visual_input",
            }[args.arm]
            row_protocol = {
                "run_protocol": run_protocol,
                "unique_observed_frames": len(images),
                "condition_visual_encode_passes": encode_passes,
                "condition_visual_frame_presentations": encode_passes * len(images),
                "episode_memory_state": episode_state,
            }
            specs = []
            for query in episode["queries"]:
                gold = str(gold_by_id[str(query["id"])])
                prompt = _query_prompt(
                    query, forecast_hidden=True,
                    condition=condition, answer_mode=args.answer_mode,
                )
                spec = {"query": query, "gold": gold, "prompt": prompt}
                if args.answer_mode == "mcq":
                    options, gold_choice = _choices(
                        query, gold, episode_id=str(episode["episode_id"]),
                        condition=condition, pools=pools,
                    )
                    spec.update(options=options, gold_choice=gold_choice)
                specs.append(spec)

            if args.answer_mode == "mcq" and args.mcq_layout == "separate":
                for spec in specs:
                    started = time.perf_counter()
                    try:
                        raw_response = invoke(images, _mcq_prompt([spec]))
                        letters = _parse_mcq_letters(raw_response, 1)
                        status, error = "ok", None
                    except Exception as exc:  # retain a row so partial runs are auditable
                        raw_response = f"__ERROR__ {type(exc).__name__}: {exc}"
                        letters, status, error = [], "error", repr(exc)
                    letter = letters[0] if letters else None
                    prediction = (spec["options"]["ABCD".index(letter)]
                                  if letter and letter in "ABCD" else "")
                    score = score_answer(prediction, spec["gold"],
                                         str(spec["query"]["type"]))
                    score["choice_correct"] = letter == spec["gold_choice"]
                    rows.append({
                        "episode_id": episode["episode_id"], "condition": condition,
                        "arm": args.arm, "query_id": spec["query"]["id"],
                        "query_type": spec["query"]["type"], "question": spec["prompt"],
                        "gold": spec["gold"], "prediction": prediction,
                        "options": spec["options"], "gold_choice": spec["gold_choice"],
                        "predicted_choice": letter, "raw_response": raw_response,
                        "parse_valid": len(letters) == 1,
                        "status": status, "error": error, "score": score,
                        "elapsed_sec": round(time.perf_counter() - started, 4),
                        "observed_frames": len(images),
                        "frames_provided_to_answer": len(images) if args.arm == "icl" else 0,
                        "forecast_card_hidden": True, "answer_mode": "mcq",
                        "mcq_layout": "separate", "card_protocol": "experiment_c_visual_v2",
                        **row_protocol,
                    })
                continue

            if args.answer_mode == "mcq":
                started = time.perf_counter()
                batch_prompt = _mcq_prompt(specs)
                try:
                    raw_response = invoke(images, batch_prompt)
                    letters = _parse_mcq_letters(raw_response, len(specs))
                    status, error = "ok", None
                except Exception as exc:  # retain a row so partial runs are auditable
                    raw_response = f"__ERROR__ {type(exc).__name__}: {exc}"
                    letters, status, error = [], "error", repr(exc)
                elapsed = round(time.perf_counter() - started, 4)
                for index, spec in enumerate(specs):
                    letter = letters[index] if index < len(letters) else None
                    prediction = (spec["options"]["ABCD".index(letter)]
                                  if letter and letter in "ABCD" else "")
                    score = score_answer(prediction, spec["gold"],
                                         str(spec["query"]["type"]))
                    score["choice_correct"] = letter == spec["gold_choice"]
                    rows.append({
                        "episode_id": episode["episode_id"], "condition": condition,
                        "arm": args.arm, "query_id": spec["query"]["id"],
                        "query_type": spec["query"]["type"], "question": spec["prompt"],
                        "gold": spec["gold"], "prediction": prediction,
                        "options": spec["options"], "gold_choice": spec["gold_choice"],
                        "predicted_choice": letter, "raw_response": raw_response,
                        "parse_valid": len(letters) == len(specs),
                        "status": status, "error": error, "score": score,
                        "elapsed_sec": elapsed, "observed_frames": len(images),
                        "frames_provided_to_answer": len(images) if args.arm == "icl" else 0,
                        "forecast_card_hidden": True,
                        "answer_mode": "mcq", "mcq_layout": "batched",
                        "card_protocol": "experiment_c_visual_v2",
                        **row_protocol,
                    })
                continue

            for spec in specs:
                query, prompt, gold = spec["query"], spec["prompt"], spec["gold"]
                started = time.perf_counter()
                try:
                    if args.arm == "icl":
                        prediction = icl.answer(images, prompt, args.max_new_tokens)
                    elif args.arm == "blind":
                        prediction = icl.answer_text(prompt, args.max_new_tokens)
                    else:
                        prediction = spatial.answer(
                            prompt, use_memory=args.arm in {"spatial", "spatial_slow_only"},
                            max_new_tokens=args.max_new_tokens,
                        )
                    status, error = "ok", None
                except Exception as exc:  # retain a row so partial runs are auditable
                    prediction, status, error = f"__ERROR__ {type(exc).__name__}: {exc}", "error", repr(exc)
                score = score_answer(prediction, gold, str(query["type"]))
                score["choice_correct"] = None
                rows.append({"episode_id": episode["episode_id"], "condition": condition,
                             "arm": args.arm, "query_id": query["id"],
                             "query_type": query["type"], "question": prompt,
                             "gold": gold, "prediction": prediction, "status": status,
                             "error": error, "score": score,
                             "elapsed_sec": round(time.perf_counter() - started, 4),
                             "observed_frames": len(images),
                             "frames_provided_to_answer": len(images) if args.arm == "icl" else 0,
                             "forecast_card_hidden": True,
                             "answer_mode": "open", "card_protocol": "experiment_c_visual_v2",
                             **row_protocol})
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [row for row in rows if row["status"] == "ok"]
    if not rows:
        return {"n": 0, "ok": 0, "mean_token_f1": 0.0, "semantic_accuracy": 0.0,
                "choice_accuracy": None, "parse_rate": None,
                "exact_normalized": 0, "gold_substring": 0}
    choice_rows = [row for row in rows if row.get("answer_mode") == "mcq"]
    result = {
        "n": len(rows), "ok": len(good),
        "mean_token_f1": round(sum(row["score"]["token_f1"] for row in rows) / len(rows), 4),
        "semantic_accuracy": round(sum(row["status"] == "ok" and bool(row["score"]["semantic_correct"])
                                       for row in rows) / len(rows), 4),
        "exact_normalized": sum(row["status"] == "ok" and bool(row["score"]["exact_normalized"])
                                for row in rows),
        "gold_substring": sum(row["status"] == "ok" and bool(row["score"]["gold_substring"])
                              for row in rows),
    }
    result["choice_accuracy"] = (round(sum(row["status"] == "ok" and bool(row["score"]["choice_correct"])
                                               for row in choice_rows) / len(choice_rows), 4)
                                 if choice_rows else None)
    result["parse_rate"] = (round(sum(row["status"] == "ok" and bool(row.get("parse_valid"))
                                      for row in choice_rows)
                                  / len(choice_rows), 4) if choice_rows else None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/experiment_c/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_c_visual.jsonl"))
    parser.add_argument("--arm", choices=("icl", "blind", *SPATIAL_ARMS), required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["relocated", "same_scene_control", "filler_only_control"])
    parser.add_argument("--answer-mode", choices=("mcq", "open"), default="mcq",
                        help="strict choice scoring; open answers are diagnostic")
    parser.add_argument("--mcq-layout", choices=("separate", "batched"), default="separate",
                        help="ask one question at a time or all five in one generation")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--model-path", default="/data/quzitsix/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--spatial-checkpoint", default="/data/quzitsix/models/Spatial-TTT-nano/model.safetensors")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--local-files-only", action="store_true", dest="local_files_only",
                         default=True)
    network.add_argument("--allow-download", action="store_false", dest="local_files_only")
    args = parser.parse_args()
    if (args.episodes < 0 or args.episode_offset < 0 or args.max_frames < 0
            or args.max_side <= 0 or args.max_new_tokens <= 0):
        raise SystemExit("episode/frame limits must be nonnegative; side/token limits must be positive")
    if len(args.conditions) != len(set(args.conditions)):
        raise SystemExit("conditions must be unique")
    rows = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
        by_type[str(row["query_type"])].append(row)
    summary = {"schema": "experiment_c_visual_v2", "arm": args.arm,
               "episodes": len({row["episode_id"] for row in rows}),
               "episode_offset": args.episode_offset, "conditions": args.conditions,
               "answer_mode": args.answer_mode,
               "mcq_layout": args.mcq_layout if args.answer_mode == "mcq" else None,
               "primary_metric": "choice_accuracy" if args.answer_mode == "mcq" else "semantic_accuracy",
               "semantic_scorer": "query-specific heuristic; diagnostic only",
               "forecast_card_hidden": all(row["forecast_card_hidden"] for row in rows),
               "run_protocol": rows[0]["run_protocol"] if rows else None,
               "rows": len(rows), "ok": sum(row["status"] == "ok" for row in rows),
               "overall": _aggregate(rows),
               "by_condition": {key: _aggregate(value) for key, value in sorted(by_condition.items())},
               "by_query_type": {key: _aggregate(value) for key, value in sorted(by_type.items())}}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": str(summary_path), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
