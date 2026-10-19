#!/usr/bin/env python3
"""Evaluate an official frozen V-JEPA 2 encoder on experiment C.

This runner is deliberately a *representation probe*.  V-JEPA 2 has no text
decoder and no persistent test-time memory, so it cannot directly answer the
open-ended questions used by ``evaluate_experiment_c_visual.py``.  Instead we
encode each chronological PNG-card sequence with the official frozen encoder
and fit a separate leave-one-content-group-out kernel-ridge readout for every
query type.  A low-resolution pixel readout and a majority-class predictor are
reported with exactly the same folds.

The resulting exact-match accuracy tests whether the final latent sequence
state makes the C labels linearly recoverable.  It is not a TTT result and it
must not be described as evidence that V-JEPA 2 itself stores long-term state.

The script uses Meta's official repository source and ``vitl.pt`` checkpoint.
It never modifies either external directory.  The defaults point at the copy
already present on this server, but both paths are configurable.

Example:

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_experiment_c_jepa.py \
      --data data/experiment_c/manifest.json \
      --output runs/experiment_c_jepa.json \
      --condition relocated --device cuda:0 --dtype bfloat16 --clip-frames 16
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_experiment_c import condition_stream  # noqa: E402


DEFAULT_SOURCE_ROOT = Path("/data/leiyj23/ICRA2026/assets/vjepa2-main-full")
DEFAULT_CHECKPOINT = Path("/data/leiyj23/ICRA2026/models/vjepa2/vitl.pt")
CONDITIONS = ("relocated", "same_scene_control", "filler_only_control")
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path, pattern: str = "*.py") -> str | None:
    """Hash relative names and bytes for every matching source file."""

    if not root.is_dir():
        return None
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _input_frame_sha256(manifest_path: Path, episodes: list[dict[str, Any]]) -> str | None:
    """Hash source PNG bytes used by the relocated condition."""

    paths = [str(relative) for episode in episodes
             if episode.get("_condition") == "relocated"
             for relative in episode.get("frame_paths", [])]
    if not paths:
        return None
    digest = hashlib.sha256()
    for relative in paths:
        path = manifest_path.parent / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _validate_features(features: dict[str, torch.Tensor], expected_rows: int) -> None:
    if not features:
        raise ValueError("feature cache contains no feature arms")
    for name, feature in features.items():
        if (not torch.is_tensor(feature) or feature.ndim != 2
                or feature.shape[0] != expected_rows or feature.shape[1] <= 0
                or not bool(torch.isfinite(feature).all())):
            raise ValueError(f"invalid cached {name} feature shape or values")


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("cannot sample an empty sequence")
    if count <= 0 or count % 2:
        raise ValueError("--clip-frames must be a positive even integer")
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _letterbox(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _render_event_card(event: dict[str, Any], *, index: int, total: int) -> Image.Image:
    """Render one condition-transformed event using the C fixture vocabulary."""

    scene = str(event.get("scene", "A"))
    background = (236, 242, 250) if scene == "A" else (250, 238, 232)
    image = Image.new("RGB", (1280, 720), background)
    draw = ImageDraw.Draw(image)
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        header_font = ImageFont.truetype(font_path, 30)
        body_font = ImageFont.truetype(font_path, 42)
    except OSError:
        header_font = body_font = ImageFont.load_default()
    header = (50, 75, 105) if scene == "A" else (110, 70, 50)
    draw.rectangle((0, 0, 1280, 90), fill=header)
    draw.text(
        (35, 25),
        f"Experiment C · {scene} · event {index + 1}/{total}",
        fill="white",
        font=header_font,
    )
    lines = [f"kind: {event['kind']}", f"time: {event['time']}"]
    lines.extend(
        f"{key}: {value}"
        for key, value in event.items()
        if key not in {"kind", "time", "scene"}
    )
    y = 150
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=65):
            draw.text((60, y), wrapped, fill=(20, 25, 35), font=body_font)
            y += 62
    return image


def _condition_episode(episode: dict[str, Any], condition: str) -> dict[str, Any]:
    events, gold_by_id = condition_stream(episode, condition)
    # V2 stores the future target separately.  Filtering here also makes a
    # legacy v1 manifest obey the same no-future-observation protocol.
    events = [event for event in events if event.get("kind") != "forecast"]
    result = dict(episode)
    result["_condition"] = condition
    result["events"] = events
    result["_content_group"] = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if condition == "relocated" and episode.get("frame_paths"):
        original_events = list(episode.get("events", []))
        original_paths = list(episode.get("frame_paths", []))
        if len(original_events) != len(original_paths):
            raise ValueError(
                f"source frame_paths/events mismatch: {len(original_paths)} != {len(original_events)}"
            )
        result["frame_paths"] = [
            path for event, path in zip(original_events, original_paths)
            if event.get("kind") != "forecast"
        ]
    result["queries"] = [
        {**query, "answer": str(gold_by_id[str(query["id"])])}
        for query in episode["queries"]
    ]
    return result


def _load_sampled_images(
    manifest_path: Path,
    episode: dict[str, Any],
    clip_frames: int,
    crop_size: int,
    clip_scope: str,
) -> list[Image.Image]:
    events = list(episode.get("events", []))
    if clip_scope == "pre_forecast":
        events = [event for event in events if event.get("kind") != "forecast"]
    indices = _uniform_indices(len(events), clip_frames)
    images = []
    for index in indices:
        if episode.get("_condition") == "relocated":
            paths = list(episode.get("frame_paths", []))
            if len(paths) != len(events):
                raise ValueError(
                    f"relocated frame_paths/events mismatch: {len(paths)} != {len(events)}"
                )
            with Image.open(manifest_path.parent / paths[index]) as source:
                card = source.convert("RGB")
        else:
            # The manifest stores files for the relocated stream.  Matched
            # control streams are derived by condition_stream and rendered in
            # memory with the same card vocabulary.
            card = _render_event_card(events[index], index=index, total=len(events))
        images.append(_letterbox(card, crop_size))
    return images


def _video_tensor(images: Iterable[Image.Image]) -> torch.Tensor:
    frames = []
    for image in images:
        array = np.asarray(image, dtype=np.float32).copy()
        frame = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
        frames.append((frame - IMAGENET_MEAN) / IMAGENET_STD)
    # Official encoder layout is [batch, channels, time, height, width].
    return torch.stack(frames, dim=1).unsqueeze(0)


def _pixel_feature(images: Iterable[Image.Image], size: int = 16) -> torch.Tensor:
    frames = []
    for image in images:
        small = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(small, dtype=np.float32).copy()
        frames.append(torch.from_numpy(array).div_(255.0).flatten())
    return torch.cat(frames).float()


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


class OfficialVJEPA2Encoder:
    """Minimal loader for Meta's official V-JEPA 2 ViT-L encoder."""

    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint: Path,
        device: str,
        dtype: str,
        crop_size: int,
    ) -> None:
        if not source_root.is_dir():
            raise FileNotFoundError(f"official V-JEPA 2 source not found: {source_root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"official V-JEPA 2 checkpoint not found: {checkpoint}")
        if crop_size != 256:
            raise ValueError("the official V-JEPA 2 ViT-L checkpoint expects --crop-size 256")

        source_text = str(source_root.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from src.models.vision_transformer import vit_large  # type: ignore

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        model = vit_large(
            patch_size=16,
            img_size=(crop_size, crop_size),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
        checkpoint_data = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        if "target_encoder" not in checkpoint_data:
            raise KeyError("checkpoint has no target_encoder; this is not the expected vitl.pt")
        incompatible = model.load_state_dict(
            _clean_state_dict(checkpoint_data["target_encoder"]), strict=False
        )
        # RoPE models have no learned positional embedding.  Other missing or
        # unexpected parameters would make the checkpoint provenance suspect.
        missing = [name for name in incompatible.missing_keys if name != "pos_embed"]
        unexpected = [name for name in incompatible.unexpected_keys if name != "pos_embed"]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint/model mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        del checkpoint_data
        gc.collect()
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.checkpoint_size = checkpoint.stat().st_size

    @torch.inference_mode()
    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        clip = clip.to(device=self.device, dtype=self.dtype)
        hidden = self.model(clip)
        frames = clip.shape[2] // 2
        spatial = hidden.shape[1] // frames
        if spatial * frames != hidden.shape[1]:
            raise RuntimeError(f"unexpected encoder output shape: {tuple(hidden.shape)}")
        # Preserve chronological information: spatially pool every tubelet and
        # concatenate the temporal bins instead of averaging the whole video.
        temporal = hidden.reshape(hidden.shape[0], frames, spatial, hidden.shape[-1]).mean(dim=2)
        feature = temporal.flatten(1).float().cpu()[0]
        return F.normalize(feature, dim=0)


def _manifest_fingerprint(
    manifest_path: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    source_inputs = {
        "source_root": str(args.source_root.resolve()),
        "source_python_tree_sha256": _sha256_tree(args.source_root / "src"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size": args.checkpoint.stat().st_size if args.checkpoint.exists() else None,
        "checkpoint_sha256": _sha256_file(args.checkpoint) if args.checkpoint.exists() else None,
    } if not args.pixel_only else {
        "source_root": None,
        "source_python_tree_sha256": None,
        "checkpoint": None,
        "checkpoint_size": None,
        "checkpoint_sha256": None,
    }
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    payload = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "episodes": [item["episode_id"] for item in manifest["episodes"][: args.episodes or None]],
        "input_frame_sha256": _input_frame_sha256(manifest_path, manifest["episodes"]),
        "renderer_font_sha256": _sha256_file(font_path) if font_path.is_file() else None,
        "clip_frames": args.clip_frames,
        "clip_scope": args.clip_scope,
        "condition": args.condition,
        "crop_size": args.crop_size,
        "dtype": args.dtype,
        **source_inputs,
        "runner_sha256": _sha256_file(Path(__file__)),
        "feature_extraction_revision": 3,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return fingerprint, payload


def _extract_features(
    manifest_path: Path,
    episodes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    fingerprint, fingerprint_inputs = _manifest_fingerprint(
        manifest_path, {"episodes": episodes}, args
    )
    if args.feature_cache and args.feature_cache.is_file() and not args.refresh_cache:
        cached = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
        has_requested_features = args.pixel_only or "vjepa2" in cached.get("features", {})
        if cached.get("fingerprint") == fingerprint and has_requested_features:
            _validate_features(cached["features"], len(episodes))
            if args.pixel_only:
                return {"pixel": cached["features"]["pixel"]}, {
                    "backend": "pixel_cache",
                    "source_cache": str(args.feature_cache),
                    "feature_cache_fingerprint": fingerprint,
                    "fingerprint_inputs": fingerprint_inputs,
                }
            encoder_meta = dict(cached["encoder"])
            encoder_meta.update({
                "feature_cache_fingerprint": fingerprint,
                "fingerprint_inputs": fingerprint_inputs,
            })
            return cached["features"], encoder_meta

    encoder = None
    if not args.pixel_only:
        encoder = OfficialVJEPA2Encoder(
            source_root=args.source_root,
            checkpoint=args.checkpoint,
            device=args.device,
            dtype=args.dtype,
            crop_size=args.crop_size,
        )
    jepa_features, pixel_features = [], []
    started = time.perf_counter()
    model_cuda_bytes = 0
    if encoder is not None and encoder.device.type == "cuda":
        model_cuda_bytes = torch.cuda.memory_allocated(encoder.device)
        torch.cuda.reset_peak_memory_stats(encoder.device)
    for index, episode in enumerate(episodes, start=1):
        images = _load_sampled_images(
            manifest_path,
            episode,
            args.clip_frames,
            args.crop_size,
            args.clip_scope,
        )
        pixel_features.append(_pixel_feature(images))
        if encoder is not None:
            jepa_features.append(encoder.encode(_video_tensor(images)))
        print(
            json.dumps(
                {"encoded": index, "total": len(episodes), "episode_id": episode["episode_id"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    features = {"pixel": torch.stack(pixel_features)}
    encoder_meta: dict[str, Any] = {
        "backend": "pixel_only" if encoder is None else "official_vjepa2",
        "elapsed_sec": round(elapsed, 4),
        "feature_cache_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
    }
    if encoder is not None:
        features["vjepa2"] = torch.stack(jepa_features)
        peak = torch.cuda.max_memory_allocated(encoder.device) if encoder.device.type == "cuda" else 0
        encoder_meta.update(
            {
                "architecture": "V-JEPA 2 ViT-L/16, FPC64-256",
                "parameter_count": encoder.parameter_count,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_bytes": encoder.checkpoint_size,
                "source_root": str(args.source_root.resolve()),
                "dtype": args.dtype,
                "device": args.device,
                "model_cuda_allocated_bytes": model_cuda_bytes,
                "peak_cuda_allocated_bytes": peak,
                "activation_over_model_bytes": max(0, peak - model_cuda_bytes),
            }
        )
    _validate_features(features, len(episodes))
    if args.feature_cache:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"fingerprint": fingerprint, "features": features, "encoder": encoder_meta},
            args.feature_cache,
        )
    return features, encoder_meta


def _episode_answers(episodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        by_type = {str(query["type"]): str(query["answer"]) for query in episode["queries"]}
        if len(by_type) != len(episode["queries"]):
            raise ValueError(f"episode {episode['episode_id']} repeats a query type")
        for query_type, answer in by_type.items():
            result[query_type].append(answer)
    lengths = {len(values) for values in result.values()}
    if lengths != {len(episodes)}:
        raise ValueError("every episode must contain the same query types")
    return dict(result)


def _training_indices(groups: list[str], held_out: int) -> list[int]:
    indices = [index for index, group in enumerate(groups) if group != groups[held_out]]
    if not indices:
        raise ValueError("content-group holdout left no training episodes")
    return indices


def _kernel_prediction(
    features: torch.Tensor,
    labels: list[str],
    groups: list[str],
    held_out: int,
    ridge: float,
) -> tuple[str, bool]:
    train_indices = _training_indices(groups, held_out)
    train_labels = [labels[index] for index in train_indices]
    classes = sorted(set(train_labels))
    train = features[train_indices].float()
    test = features[held_out].float()
    mean = train.mean(dim=0, keepdim=True)
    train = F.normalize(train - mean, dim=1)
    test = F.normalize(test - mean[0], dim=0)
    targets = torch.zeros(len(train_indices), len(classes))
    for row, label in enumerate(train_labels):
        targets[row, classes.index(label)] = 1.0
    kernel = train @ train.T
    coefficients = torch.linalg.solve(
        kernel + ridge * torch.eye(len(train_indices)), targets
    )
    scores = (test @ train.T) @ coefficients
    return classes[int(scores.argmax())], labels[held_out] in classes


def _majority_prediction(
    labels: list[str], groups: list[str], held_out: int
) -> tuple[str, bool]:
    train = [labels[index] for index in _training_indices(groups, held_out)]
    counts = Counter(train)
    prediction = sorted(counts, key=lambda label: (-counts[label], label))[0]
    return prediction, labels[held_out] in counts


def _evaluate_readouts(
    episodes: list[dict[str, Any]],
    features: dict[str, torch.Tensor],
    ridge: float,
    condition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_by_type = _episode_answers(episodes)
    content_groups = [str(episode["_content_group"]) for episode in episodes]
    rows: list[dict[str, Any]] = []
    arms = list(features) + ["majority"]
    for arm in arms:
        for query_type, labels in labels_by_type.items():
            for held_out, episode in enumerate(episodes):
                if arm == "majority":
                    prediction, candidate_seen = _majority_prediction(
                        labels, content_groups, held_out
                    )
                else:
                    prediction, candidate_seen = _kernel_prediction(
                        features[arm], labels, content_groups, held_out, ridge
                    )
                rows.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "episode_id": episode["episode_id"],
                        "content_group": content_groups[held_out],
                        "training_episode_count": len(
                            _training_indices(content_groups, held_out)
                        ),
                        "query_type": query_type,
                        "gold": labels[held_out],
                        "prediction": prediction,
                        "correct": prediction == labels[held_out],
                        "gold_seen_in_training_fold": candidate_seen,
                    }
                )

    summary: dict[str, Any] = {
        "by_arm": {},
        "by_arm_and_query_type": {},
        "expected_baselines": {},
    }
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        seen = [row for row in arm_rows if row["gold_seen_in_training_fold"]]
        summary["by_arm"][arm] = {
            "n": len(arm_rows),
            "accuracy": round(sum(row["correct"] for row in arm_rows) / len(arm_rows), 4),
            "seen_n": len(seen),
            "seen_accuracy": round(sum(row["correct"] for row in seen) / len(seen), 4) if seen else None,
        }
        for query_type in labels_by_type:
            group = [row for row in arm_rows if row["query_type"] == query_type]
            seen_group = [row for row in group if row["gold_seen_in_training_fold"]]
            summary["by_arm_and_query_type"][f"{arm}/{query_type}"] = {
                "n": len(group),
                "accuracy": round(sum(row["correct"] for row in group) / len(group), 4),
                "seen_n": len(seen_group),
                "seen_accuracy": round(
                    sum(row["correct"] for row in seen_group) / len(seen_group), 4
                ) if seen_group else None,
            }

    # A sampled training-fold class prior is more informative than argmax on
    # this balanced fixture: leave-one-out makes the held-out class the unique
    # minority for several query types, so deterministic majority scores zero.
    prior_by_type = {}
    for query_type, labels in labels_by_type.items():
        probabilities = []
        for held_out, gold in enumerate(labels):
            train = [
                labels[index]
                for index in _training_indices(content_groups, held_out)
            ]
            probabilities.append(Counter(train)[gold] / len(train))
        prior_by_type[query_type] = round(sum(probabilities) / len(probabilities), 4)
    summary["expected_baselines"]["sample_training_fold_label_prior"] = {
        "expected_accuracy": round(sum(prior_by_type.values()) / len(prior_by_type), 4),
        "by_query_type": prior_by_type,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/experiment_c/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_c_jepa.json"))
    parser.add_argument("--feature-cache", type=Path, default=Path("runs/experiment_c_jepa_features.pt"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--episodes", type=int, default=0, help="0 uses every episode")
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--clip-scope", choices=("all", "pre_forecast"), default="all")
    parser.add_argument("--condition", choices=CONDITIONS, default="relocated")
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--pixel-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    if args.episodes < 0:
        raise SystemExit("--episodes must be nonnegative")
    if args.ridge <= 0:
        raise SystemExit("--ridge must be positive")
    _uniform_indices(2, args.clip_frames)  # validate before loading the model

    manifest = json.loads(args.data.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {
        "experiment_c_relocation_v1",
        "experiment_c_relocation_v2",
    }:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    source_episodes = list(manifest.get("episodes", []))[: args.episodes or None]
    episodes = [_condition_episode(episode, args.condition) for episode in source_episodes]
    if len(episodes) < 3:
        raise ValueError("leave-one-episode-out evaluation needs at least three episodes")
    observed_counts = {len(episode["events"]) for episode in episodes}
    if len(observed_counts) != 1:
        raise ValueError(f"condition has inconsistent observed frame counts: {observed_counts}")

    features, encoder_meta = _extract_features(args.data, episodes, args)
    rows, metrics = _evaluate_readouts(episodes, features, args.ridge, args.condition)
    result = {
        "schema": "experiment_c_jepa_probe_v2",
        "protocol": {
            "source_schema": manifest.get("schema"),
            "episodes": len(episodes),
            "condition": args.condition,
            "content_groups": len({episode["_content_group"] for episode in episodes}),
            "observed_frames_per_episode": sorted(observed_counts),
            "clip_frames": args.clip_frames,
            "clip_scope": args.clip_scope,
            "future_event_observed": False,
            "crop_size": args.crop_size,
            "readout": "leave-one-content-group-out centered kernel ridge",
            "ridge": args.ridge,
            "label_access": "training-fold labels only",
            "interpretation": "frozen representation probe; no TTT and no persistent JEPA memory",
        },
        "encoder": encoder_meta,
        "metrics": metrics,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **metrics["by_arm"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
