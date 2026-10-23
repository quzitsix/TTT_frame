"""Past-only per-image visual features for the forecasting residual preflight.

Images in a batch have independent attention boundaries. The temporal patch
of each still image duplicates that same image; no later stream frame enters
an earlier representation. The random projection and token normalization
have no fitted dataset statistics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_frame.forecast_bench import ForecastCase, FrameSample


@dataclass(frozen=True)
class FeatureConfig:
    sample_fps: float = 2.0
    max_side: int = 256
    projection_dim: int = 256
    projection_seed: int = 0
    pool_shape: tuple[int, int] = (2, 4)
    dtype: str = "bfloat16"
    context_tail_seconds: float | None = None
    limit_frames: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_fps) or not 0 < self.sample_fps <= 2:
            raise ValueError("sample_fps must be finite and in (0, 2]")
        if self.max_side < 64 or self.max_side % 32:
            raise ValueError("max_side must be a multiple of 32 and at least 64")
        if self.projection_dim != 256 or self.pool_shape != (2, 4):
            raise ValueError("this cache format fixes pool_shape=(2, 4) and projection_dim=256")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("unsupported encoder dtype")
        if self.context_tail_seconds is not None and (
            not math.isfinite(self.context_tail_seconds)
            or not 0 <= self.context_tail_seconds <= 1800
        ):
            raise ValueError("context_tail_seconds must lie in [0, 1800]")
        if isinstance(self.limit_frames, bool) or not isinstance(self.limit_frames, int):
            raise ValueError("limit_frames must be a nonnegative integer")
        if self.limit_frames < 0:
            raise ValueError("limit_frames must be a nonnegative integer")


def sampling_plan(case: ForecastCase, config: FeatureConfig) -> tuple[FrameSample, ...]:
    """Use a single regular grid across the context/recent boundary."""
    count = case.prefix_seconds * config.sample_fps
    context_count = case.context_seconds * config.sample_fps
    if not math.isclose(count, round(count)) or not math.isclose(
        context_count, round(context_count)
    ):
        raise ValueError("sample_fps must align with both clip duration boundaries")
    start = 0.0 if config.context_tail_seconds is None else (
        case.context_seconds - config.context_tail_seconds
    )
    first_step = math.ceil(start * config.sample_fps)
    plan = tuple(
        FrameSample(
            stream_step=step,
            prefix_seconds=step / config.sample_fps,
            video_seconds=(step - (context_count if step >= context_count else 0))
                / config.sample_fps,
            role="context" if step < context_count else "recent",
            path=case.context_path if step < context_count else case.recent_path,
        )
        for step in range(first_step, round(count))
    )
    return plan[:config.limit_frames] if config.limit_frames else plan


def causal_video_frames(
    path: Path, requested_seconds: list[float], *, max_side: int,
) -> Iterator[tuple[object, float]]:
    """Yield the latest decoded frame with PTS <= each requested time.

Decoding may need one lookahead frame to locate a boundary; only the previous
frame is converted and returned. Stale frames across gaps/short files raise.
"""
    import av

    if not requested_seconds:
        return
    if any(not math.isfinite(t) or t < 0 for t in requested_seconds):
        raise ValueError("requested times must be finite and nonnegative")
    if any(b <= a for a, b in zip(requested_seconds, requested_seconds[1:])):
        raise ValueError("requested times must be strictly increasing")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_count = 2
        origin = float((stream.start_time or 0) * stream.time_base)
        if not stream.average_rate or float(stream.average_rate) <= 0:
            raise ValueError("a known source frame rate is required")
        max_age = 1 / float(stream.average_rate) + 1e-6
        if requested_seconds[0] > 0:
            container.seek(
                int((origin + requested_seconds[0]) / float(stream.time_base)),
                stream=stream, backward=True, any_frame=False,
            )
        decoded = iter(container.decode(stream))
        lookahead = next(decoded, None)
        previous = None
        previous_time = None
        last_seen = -math.inf
        for target in requested_seconds:
            while lookahead is not None:
                if lookahead.pts is None:
                    raise ValueError("decoded frame has no presentation timestamp")
                timestamp = float(lookahead.pts * stream.time_base) - origin
                if timestamp < last_seen - 1e-7:
                    raise ValueError("nonmonotonic video presentation times")
                if timestamp > target + 1e-9:
                    break
                last_seen = timestamp
                previous, previous_time = lookahead, timestamp
                lookahead = next(decoded, None)
            if previous is None or previous_time is None:
                raise ValueError(f"no observed frame at or before {target} in {path}")
            if target - previous_time > max_age:
                raise ValueError(f"source gap or short video at {target} in {path}")
            picture = previous.to_image().convert("RGB")
            picture.thumbnail((max_side, max_side))
            yield picture, previous_time


def orthogonal_projection(input_dim: int, output_dim: int, seed: int) -> Tensor:
    if input_dim < output_dim or output_dim < 1:
        raise ValueError("orthogonal projection requires input_dim >= output_dim > 0")
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    # Resolve QR sign ambiguity for a stable CPU-generated projection.
    signs = torch.where(r.diagonal() < 0, -1.0, 1.0)
    return q * signs


def pool_project_image(
    tokens: Tensor, height: int, width: int, projection: Tensor,
    pool_shape: tuple[int, int] = (2, 4),
) -> Tensor:
    if tokens.ndim != 2 or tokens.shape != (height * width, projection.shape[0]):
        raise ValueError("image token dimensions do not match its merged spatial grid")
    spatial = tokens.float().reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(spatial, pool_shape).squeeze(0).flatten(1).T
    projected = pooled @ projection.to(device=tokens.device, dtype=torch.float32)
    result = F.layer_norm(projected, (projection.shape[1],), eps=1e-5)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite frozen visual features")
    return result


class QwenImageFeatures:
    """Load only the local Qwen3-VL visual tower, never the language decoder."""

    def __init__(self, model_path: Path, config: FeatureConfig, device: str = "cpu"):
        from safetensors import safe_open
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionModel, Qwen3VLVisionRotaryEmbedding,
        )

        self.config, self.device = config, torch.device(device)
        self.dtype = getattr(torch, config.dtype)
        model_path = Path(model_path).resolve()
        full_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if full_config.model_type != "qwen3_vl":
            raise ValueError("encoder requires a local Qwen3-VL checkpoint")
        vision_config = full_config.vision_config
        vision_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3VLVisionModel(vision_config)
        # Rotary frequencies are nonpersistent buffers and absent from weights.
        model.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            (vision_config.hidden_size // vision_config.num_heads) // 2
        )
        expected = set(model.state_dict())
        state, digest = {}, hashlib.sha256()
        for filename in ("config.json", "preprocessor_config.json"):
            digest.update(filename.encode())
            digest.update((model_path / filename).read_bytes())
        prefix = "model.visual."
        for path in sorted(model_path.glob("model*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as source:
                for name in sorted(source.keys()):
                    if not name.startswith(prefix):
                        continue
                    local = name[len(prefix):]
                    if local in state:
                        raise ValueError(f"duplicate visual parameter {local}")
                    raw = source.get_tensor(name)
                    digest.update(name.encode())
                    digest.update(str((tuple(raw.shape), raw.dtype)).encode())
                    digest.update(raw.contiguous().view(torch.uint8).numpy())
                    state[local] = raw.to(dtype=self.dtype)
        if set(state) != expected:
            raise ValueError(f"visual checkpoint mismatch: missing={expected-set(state)}, "
                             f"extra={set(state)-expected}")
        model.load_state_dict(state, assign=True, strict=True)
        self.model = model.to(device=self.device).eval().requires_grad_(False)
        del state
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=True,
            size={"shortest_edge": 32 * 32, "longest_edge": config.max_side**2},
        )
        self.projection = orthogonal_projection(
            vision_config.out_hidden_size, config.projection_dim, config.projection_seed
        ).to(self.device)
        self.model_fingerprint = {
            "kind": "visual_tensor_content_plus_config_sha256",
            "sha256": digest.hexdigest(), "visual_parameters": len(expected),
        }

    @torch.inference_mode()
    def encode(self, images: list) -> Tensor:
        batch = self.processor(images=images, return_tensors="pt")
        # HF's SDPA visual path uses these sizes in Python loops and .tolist()
        # in every block. Keep metadata on CPU to avoid hundreds of CUDA
        # synchronization points; pixel tensors and model arithmetic stay on GPU.
        grid = batch["image_grid_thw"].cpu()
        if bool((grid[:, 0] != 1).any()):
            raise ValueError("every image must have an independent temporal grid of one")
        tokens, _ = self.model(
            batch["pixel_values"].to(device=self.device, dtype=self.dtype), grid_thw=grid
        )
        merge = self.model.spatial_merge_size
        offset, output = 0, []
        for _, h, w in grid.tolist():
            height, width = h // merge, w // merge
            length = height * width
            output.append(pool_project_image(
                tokens[offset:offset + length], height, width, self.projection,
                self.config.pool_shape,
            ))
            offset += length
        if offset != tokens.shape[0]:
            raise ValueError("visual output does not match image grids")
        return torch.stack(output).cpu()


def cache_case(
    case: ForecastCase, encoder: QwenImageFeatures, *, input_manifest_sha256: str,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = encoder.config
    plan = sampling_plan(case, config)
    if not plan:
        raise ValueError("empty sampling plan")
    all_features, actual_times = [], []
    processed = 0
    for role in ("context", "recent"):
        samples = [sample for sample in plan if sample.role == role]
        if not samples:
            continue
        offset = 0.0 if role == "context" else case.context_seconds
        batch = []
        for image, actual in causal_video_frames(
            samples[0].path, [sample.video_seconds for sample in samples],
            max_side=config.max_side,
        ):
            batch.append(image)
            actual_times.append(offset + actual)
            if len(batch) == batch_size:
                all_features.append(encoder.encode(batch))
                processed += len(batch)
                if progress is not None:
                    progress(processed, len(plan))
                batch = []
        if batch:
            all_features.append(encoder.encode(batch))
            processed += len(batch)
            if progress is not None:
                progress(processed, len(plan))
    features = torch.cat(all_features).float()
    if features.shape != (len(plan), 8, 256):
        raise ValueError("encoder did not return one 8x256 representation per requested frame")
    complete = plan[0].stream_step == 0 and len(plan) == round(
        case.prefix_seconds * config.sample_fps
    )
    return {
        "format_version": 1, "kind": "bench_v0_causal_visual_features",
        "sample_id": case.id, "features": features,
        "requested_prefix_seconds": torch.tensor(
            [sample.prefix_seconds for sample in plan], dtype=torch.float64
        ),
        "actual_prefix_seconds": torch.tensor(actual_times, dtype=torch.float64),
        "sample_fps": config.sample_fps, "config": asdict(config),
        "encoder": "qwen3_vl_per_image", "input_manifest_sha256": input_manifest_sha256,
        "model_fingerprint": encoder.model_fingerprint,
        "cropped": not complete, "complete_prefix": complete, "audio_used": False,
        "normalization": "per_token_parameter_free_layer_norm_eps_1e-5",
        "source_media": {
            role: {"path": str(path), "size_bytes": path.stat().st_size,
                   "mtime_ns": path.stat().st_mtime_ns}
            for role, path in (("context", case.context_path), ("recent", case.recent_path))
        },
    }
