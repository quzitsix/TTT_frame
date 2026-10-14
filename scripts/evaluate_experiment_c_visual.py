#!/usr/bin/env python3
"""Quick visual follow-up for the synthetic experiment-C manifest.

The symbolic runner is the primary mechanism result.  This optional script
feeds the generated event cards through native Qwen3-VL and (when requested)
Spatial-TTT, so it also exposes OCR/card-reading and model-readout failures.
It deliberately defaults to two episodes; the cards are text-heavy synthetic
stimuli and are not a substitute for a real first-person video benchmark.

Examples::

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_experiment_c_visual.py \
      --data data/experiment_c/manifest.json --episodes 2 --arm icl \
      --device cuda:2

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_experiment_c_visual.py \
      --data data/experiment_c/manifest.json --episodes 2 --arm spatial \
      --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
      --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
      --device cuda:2
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory  # noqa: E402


def _tokens(value: str) -> list[str]:
    value = str(value or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"[a-z0-9]+", value)


def score_answer(prediction: str, gold: str) -> dict[str, Any]:
    predicted, target = _tokens(prediction), _tokens(gold)
    overlap = sum((Counter(predicted) & Counter(target)).values())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    ptext, gtext = " ".join(predicted), " ".join(target)
    return {"exact_normalized": ptext == gtext,
            "gold_substring": bool(gtext) and gtext in ptext,
            "token_f1": round(f1, 4)}


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
             "Frames are chronological. Read only the visible event cards. "
             "Answer briefly.\n" + question},
        ]}], tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], videos=[array], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt")
        batch = self._move(batch)
        output = self.model.generate(**batch, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
        continuation = output[:, batch["input_ids"].shape[1]:]
        return self.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.data.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])[:args.episodes or None]
    if not episodes:
        raise ValueError("manifest contains no episodes")
    # Loading one large VLM twice can exhaust a 24 GB card.  Run arms one at a
    # time and let the caller invoke this command again for the other arm.
    icl = None
    spatial = None
    if args.arm == "icl":
        icl = NativeICL(args.model_path, args.device, args.dtype, args.local_files_only)
    elif args.arm == "spatial":
        config = SpatialVideoConfig(model_path=args.model_path, device=args.device,
                                    dtype=args.dtype, local_files_only=args.local_files_only,
                                    spatial_checkpoint=args.spatial_checkpoint,
                                    chunk_seconds=3600.0, frames_per_chunk=4096,
                                    max_side=args.max_side, max_new_tokens=args.max_new_tokens)
        spatial = SpatialVideoMemory(config)
    rows = []
    for episode in episodes:
        images = _uniform(_load_images(args.data, episode, args.max_side), args.max_frames)
        if not images:
            raise ValueError(f"episode {episode['episode_id']} has no frame cards")
        if spatial is not None:
            spatial.reset()
            spatial.ingest_frames(images, [float(index) for index in range(len(images))])
            spatial.finish_ingest()
        for query in episode["queries"]:
            started = time.perf_counter()
            try:
                if icl is not None:
                    prediction = icl.answer(images, query["question"], args.max_new_tokens)
                else:
                    prediction = spatial.answer(query["question"], max_new_tokens=args.max_new_tokens)
                status, error = "ok", None
            except Exception as exc:  # retain a row so partial runs are auditable
                prediction, status, error = f"__ERROR__ {type(exc).__name__}: {exc}", "error", repr(exc)
            rows.append({"episode_id": episode["episode_id"], "arm": args.arm,
                         "query_id": query["id"], "query_type": query["type"],
                         "question": query["question"], "gold": query["answer"],
                         "prediction": prediction, "status": status,
                         "score": score_answer(prediction, query["answer"]),
                         "elapsed_sec": round(time.perf_counter() - started, 4),
                         "frames": len(images), "card_protocol": "relocated_synthetic"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/experiment_c/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_c_visual.jsonl"))
    parser.add_argument("--arm", choices=("icl", "spatial"), required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--model-path", default="/data/quzitsix/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--spatial-checkpoint", default="/data/quzitsix/models/Spatial-TTT-nano/model.safetensors")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()
    rows = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    good = [row for row in rows if row["status"] == "ok"]
    summary = {"arm": args.arm, "episodes": args.episodes, "rows": len(rows),
               "ok": len(good), "mean_token_f1": round(sum(row["score"]["token_f1"] for row in good) / len(good), 4) if good else 0.0,
               "exact_normalized": sum(row["score"]["exact_normalized"] for row in good),
               "gold_substring": sum(row["score"]["gold_substring"] for row in good)}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": str(summary_path), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
