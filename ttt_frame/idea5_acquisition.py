"""Causal thumbnail acquisition preflight, not the trained Idea5 selector.

One decision follows six newly observed frames. All variants use identical
multi-video packing; the fixed finite prefix schedule defines four budget cells.
The motion policy sees only current/past thumbnails and never the question.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Sequence

import numpy as np

ARMS = ("low112", "mid224", "high448", "uniform_high", "random_high", "motion_high")
CELLS = ((0, 4), (4, 8), (8, 12), (12, 15))


def thumbnail(image, side: int):
    if side not in (112, 224, 448):
        raise ValueError("unsupported resolution")
    result = image.convert("RGB").copy()
    result.thumbnail((side, side))
    return result


def chunk_motion(images: Sequence) -> float:
    if len(images) != 6:
        raise ValueError("expected six current/past thumbnails")
    # Scoring is deliberately bounded to current low-resolution evidence.
    arrays = [np.asarray(thumbnail(im, 112).convert("L").resize((32, 32)), dtype=np.float32) / 255
              for im in images]
    return float(np.mean([np.abs(b-a).mean() for a, b in zip(arrays, arrays[1:])]))


class CausalMotionPolicy:
    """One high chunk per public schedule cell, using current/past motion only.

    Threshold is fixed before evaluation. The last chunk is selected if no
    earlier chunk cleared it. Knowledge of cell endpoints is budget information,
    not an observation from future frames. No prefix-wide ranking is performed.
    """
    def __init__(self, threshold: float = 0.1):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid threshold")
        self.threshold = threshold
        self.next_index = 0
        self.selected_cells: set[int] = set()

    def step(self, index: int, motion: float) -> bool:
        if index != self.next_index or not 0 <= index < 15:
            raise ValueError("chunks must arrive in chronological order")
        if not math.isfinite(motion) or motion < 0:
            raise ValueError("invalid motion")
        self.next_index += 1
        cell = next(i for i, (start, stop) in enumerate(CELLS) if start <= index < stop)
        selected = cell not in self.selected_cells and (
            motion >= self.threshold or index == CELLS[cell][1] - 1)
        if selected:
            self.selected_cells.add(cell)
        return selected


def resolution_schedules(sample_id: str, motions: Sequence[float], seed: int = 0) -> dict[str, list[int]]:
    if len(motions) != 15 or any(not math.isfinite(x) or x < 0 for x in motions):
        raise ValueError("15 finite nonnegative chronological motions required")
    stable_seed = int.from_bytes(hashlib.sha256(f"{seed}/{sample_id}".encode()).digest()[:8], "little")
    generator = random.Random(stable_seed)
    random_indices = {generator.randrange(start, stop) for start, stop in CELLS}
    uniform_indices = {start for start, _ in CELLS}
    policy = CausalMotionPolicy()
    motion_indices = {i for i, value in enumerate(motions) if policy.step(i, value)}
    return {
        "low112": [112] * 15, "mid224": [224] * 15, "high448": [448] * 15,
        "uniform_high": [448 if i in uniform_indices else 112 for i in range(15)],
        "random_high": [448 if i in random_indices else 112 for i in range(15)],
        "motion_high": [448 if i in motion_indices else 112 for i in range(15)],
    }


def build_batch(engine, chunks, times, resolutions, question):
    from transformers.video_utils import VideoMetadata
    if len(chunks) != 15 or len(times) != 15 or len(resolutions) != 15:
        raise ValueError("expected 15 chronological chunks")
    videos, metadata = [], []
    for pictures, timestamps, side in zip(chunks, times, resolutions):
        if len(pictures) != 6 or len(timestamps) != 6:
            raise ValueError("six real frames per chunk; no padding")
        indices = [int(round(t * 1000)) for t in timestamps]
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("strictly increasing timestamps required")
        videos.append(np.stack([np.asarray(thumbnail(im, side)) for im in pictures]))
        metadata.append(VideoMetadata(total_num_frames=max(indices[-1] + 1, 6),
                                      fps=1000.0, frames_indices=indices))
    prompt = engine.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"} for _ in chunks]
          + [{"type": "text", "text": question}]}], tokenize=False, add_generation_prompt=True)
    batch = engine.processor(text=[prompt], videos=videos, video_metadata=metadata,
                             do_sample_frames=False, return_tensors="pt")
    return engine._move(batch)


def answer(engine, chunks, times, resolutions, question, tokens):
    import time
    import torch
    from scripts.evaluate_forecast_mcq import choose_label
    engine.reset()
    torch.cuda.synchronize(engine.device)
    start = time.perf_counter()
    batch = build_batch(engine, chunks, times, resolutions, question)
    torch.cuda.synchronize(engine.device)
    prepared = time.perf_counter()
    with torch.no_grad():
        output = engine.model(**batch, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize(engine.device)
    finished = time.perf_counter()
    result = choose_label(output.logits[0, -1], tokens, engine.processor.tokenizer)
    result.update(input_tokens=int(batch["input_ids"].shape[1]),
                  visual_tokens=int(batch["input_ids"].eq(engine.model.config.video_token_id).sum()),
                  video_grid_thw=batch["video_grid_thw"].tolist(),
                  preprocess_seconds=prepared-start, forward_seconds=finished-prepared,
                  elapsed_seconds=finished-start,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device))
    engine.reset()
    return result
