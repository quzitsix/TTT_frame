"""Teacher-free video -> Spatial-TTT fast weights -> parameter-only questions.

This is a reference implementation of the Spatial-TTT core with a project-specific
episode boundary. Observation KV and pixels are discarded between input clips;
questions do not write memory. A trained spatial checkpoint is needed for useful
readout: the official readout scale starts at zero when no checkpoint is supplied.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
import time

import torch

from ttt_frame.spatial_model import SpatialModelConfig, SpatialQwenMemory, load_official_checkpoint
from ttt_frame.teacher_bridge import (canonicalize_analysis, prepare_teacher_packet,
                                      validate_teacher_analysis)
from ttt_frame.video import iter_video_chunks

log = logging.getLogger(__name__)


@dataclass
class SpatialVideoConfig:
    model_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    spatial_checkpoint: str | None = None
    chunk_seconds: float = 4.0
    frames_per_chunk: int = 8
    max_side: int = 448
    max_chunks: int = 0
    max_new_tokens: int = 128
    spatial: SpatialModelConfig = field(default_factory=SpatialModelConfig)

    def __post_init__(self):
        if isinstance(self.spatial, dict):
            self.spatial = SpatialModelConfig(**self.spatial)
        if not self.model_path or self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("model_path is required; dtype must be float32 or bfloat16")
        if min(self.chunk_seconds, self.frames_per_chunk, self.max_side, self.max_new_tokens) <= 0 or self.max_chunks < 0:
            raise ValueError("invalid video or generation budget")


def _file_hash(path):
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SpatialVideoMemory:
    """One continuous environment: write -> ask -> write -> save -> resume.

    Persisted content is numerical model state only, including parameter-shaped
    update momentum. No observation text, frame embeddings or history KV survive
    a public ingestion call. New independent environments must call reset().
    """

    def __init__(self, config: SpatialVideoConfig, *, model=None, processor=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        if self.device.type == "cuda" and self.dtype == torch.bfloat16:
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("GPU does not support bfloat16; use float32")
        self.processor = processor or AutoProcessor.from_pretrained(
            config.model_path, local_files_only=config.local_files_only,
        )
        self.model = model or AutoModelForImageTextToText.from_pretrained(
            config.model_path, local_files_only=config.local_files_only, dtype=self.dtype,
            device_map={"": str(self.device)}, attn_implementation="sdpa",
        )
        fingerprint = self.model.config.to_dict()
        for key in ("_name_or_path", "transformers_version", "_commit_hash"):
            fingerprint.pop(key, None)
        self.base_fingerprint = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()
        self.controller = SpatialQwenMemory(self.model, config.spatial)
        self.spatial_checkpoint_hash = None
        if config.spatial_checkpoint:
            load_official_checkpoint(self.controller, config.spatial_checkpoint)
            self.spatial_checkpoint_hash = _file_hash(config.spatial_checkpoint)
        else:
            log.warning("No trained Spatial-TTT checkpoint: this is a mechanism initialization. "
                        "The default zero readout scale cannot retrieve the written fast weights.")
        self.model.eval()
        self.reset()

    def _clear_generation_state(self):
        for module in self.model.modules():
            if "rope_deltas" in vars(module):
                module.rope_deltas = None

    def reset(self):
        self.controller.reset()
        self.position_offset = 0
        self.stats = dict(sessions=0, chunks=0, frames=0, visual_tokens=0, input_tokens=0,
                          ingest_seconds=0.0)
        self.state = "ready"
        self._clear_generation_state()

    def _check_ready(self):
        if self.state == "failed":
            raise RuntimeError("failed ingestion cannot be queried or resumed; reset or load a valid memory")

    def _move(self, batch):
        return {key: value.to(self.device, dtype=self.dtype if value.is_floating_point() else value.dtype)
                for key, value in batch.items() if torch.is_tensor(value)}

    def _encode_video(self, images, timestamps):
        import numpy as np
        from transformers.video_utils import VideoMetadata

        # Images are already sampled by PyAV. Disable the processor's second sampling pass.
        frames = np.stack([np.asarray(im.convert("RGB")) for im in images])
        indices = [int(round(t * 1000)) for t in timestamps]
        metadata = VideoMetadata(total_num_frames=max(indices[-1] + 1, len(images)),
                                 fps=1000.0, frames_indices=indices)
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": "Observe the visible environment."},
            ]}], tokenize=False, add_generation_prompt=True,
        )
        batch = self.processor(text=[prompt], videos=[frames], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt")
        if "pixel_values_videos" not in batch or "video_grid_thw" not in batch:
            raise RuntimeError("processor did not produce native video pixels and grid")
        return self._move(batch)

    @torch.no_grad()
    def ingest_frames(self, images: list, timestamps: list[float] | None = None):
        self._check_ready()
        if not images:
            raise ValueError("at least one image is required")
        timestamps = list(timestamps) if timestamps is not None else [float(i) for i in range(len(images))]
        if (len(timestamps) != len(images) or any(not 0 <= t < float("inf") for t in timestamps)
                or any(b < a for a, b in zip(timestamps, timestamps[1:]))):
            raise ValueError("timestamps must be finite, nonnegative, ordered, and match the images")
        started = time.perf_counter()
        try:
            self._clear_generation_state()
            batch = self._encode_video(images, timestamps)
            video_mask = batch["input_ids"].eq(self.model.config.video_token_id)
            if not video_mask.any():
                raise RuntimeError("native video input produced no video tokens")
            positions, _ = self.model.model.get_rope_index(
                batch["input_ids"], video_grid_thw=batch["video_grid_thw"],
                attention_mask=batch.get("attention_mask"),
            )
            positions = positions + self.position_offset
            with self.controller.context("write", video_mask=video_mask,
                                         video_grid_thw=batch["video_grid_thw"], flush=True):
                # No captions, pseudo-QA, labels, loss.backward(), or optimizer.
                output = self.model(**batch, position_ids=positions, use_cache=False, logits_to_keep=1)
            if not torch.isfinite(output.logits).all():
                raise RuntimeError("nonfinite Spatial-TTT output during ingestion")
            # The final write happens after that chunk's read. Finite logits do
            # not imply the last W/momentum update was finite (zero gate hides it).
            for layer in self.controller.layers.values():
                layer.state.validate(layer.memory.config, batch_size=1)
            self.controller.detach()
            self.position_offset = int(positions.max().item()) + 1
            report = dict(chunks=1, frames=len(images), visual_tokens=int(video_mask.sum()),
                          input_tokens=batch["input_ids"].shape[1], ingest_seconds=time.perf_counter()-started)
            for key, value in report.items():
                self.stats[key] += value
            return report
        except Exception:
            self.state = "failed"
            raise
        finally:
            self._clear_generation_state()

    def ingest_video(self, path: str | Path):
        self._check_ready()
        before = dict(self.stats)
        try:
            with closing(iter_video_chunks(path, chunk_seconds=self.config.chunk_seconds,
                                           frames_per_chunk=self.config.frames_per_chunk,
                                           max_side=self.config.max_side,
                                           max_chunks=self.config.max_chunks)) as chunks:
                for chunk in chunks:
                    self.ingest_frames(chunk.images, chunk.timestamps)
            self.stats["sessions"] += 1
        except Exception:
            self.state = "failed"
            raise
        return {key: self.stats[key] - value for key, value in before.items()}

    @torch.no_grad()
    def finish_ingest(self):
        """Commit the last pending tokens, without locking out future observations."""
        self._check_ready()
        self.controller.flush()
        self.controller.detach()
        self._clear_generation_state()
        return {"memory_kind": "spatial_fast_weights", "memory_bytes": self.controller.memory_bytes,
                "runtime_state_bytes": self.controller.runtime_state_bytes,
                "ttt_layers": list(self.controller.layers), "stats": dict(self.stats),
                "parameter_only": True, "retained_video_frames": 0, "retained_text_records": 0,
                "updates_per_layer": {str(i): layer.state.updates if layer.state else 0
                                      for i, layer in self.controller.layers.items()},
                "trained_checkpoint_loaded": self.spatial_checkpoint_hash is not None}

    @torch.no_grad()
    def answer(self, question: str, *, use_memory: bool = True, max_new_tokens: int | None = None):
        """Greedy decode with a temporary text-only KV cache and no memory writes."""
        self._check_ready()
        if not question.strip():
            raise ValueError("question must not be empty")
        budget = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        if budget <= 0:
            raise ValueError("max_new_tokens must be positive")
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": question}]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self._move(self.processor(text=[rendered], return_tensors="pt"))
        input_ids = inputs["input_ids"]
        length = input_ids.shape[1]
        mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        generated, cache = [], None
        self._clear_generation_state()
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        stopped_on_eos = False
        try:
            with self.controller.context("read" if use_memory else "base"):
                for step in range(budget):
                    offset = 0 if cache is None else length + step - 1
                    cache_position = torch.arange(offset, offset + input_ids.shape[1], device=self.device)
                    positions = (cache_position + self.position_offset)[None, None].expand(3, 1, -1)
                    outputs = self.model(input_ids=input_ids, attention_mask=mask,
                                         position_ids=positions, cache_position=cache_position,
                                         past_key_values=cache, use_cache=True, logits_to_keep=1)
                    if not torch.isfinite(outputs.logits).all():
                        raise RuntimeError("nonfinite Spatial-TTT answer logits")
                    token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
                    generated.append(token)
                    if int(token.item()) in eos_ids:
                        stopped_on_eos = True
                        break
                    cache = outputs.past_key_values
                    input_ids = token
                    mask = torch.cat((mask, torch.ones_like(token)), dim=1)
        finally:
            self._clear_generation_state()
        if generated and not stopped_on_eos and len(generated) >= budget:
            log.warning(
                "generation reached max_new_tokens=%d before EOS; output may be truncated",
                budget,
            )
        return self.processor.batch_decode(torch.cat(generated, dim=1), skip_special_tokens=True)[0].strip()

    def save(self, directory: str | Path):
        from safetensors.torch import save_file

        self.finish_ingest()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        weights = {k: v.cpu().contiguous() for k, v in self.controller.memory_state_dict().items()}
        if any("pending_" in key for key in weights):
            raise RuntimeError("pending visual tokens must be committed before saving")
        save_file(weights, str(directory / "fast_weights.safetensors"))
        save_file(self.controller.slow_state_dict(), str(directory / "spatial.safetensors"))
        metadata = dict(format_version=1, kind="spatial_fast_weights", config=asdict(self.config),
                        base_fingerprint=self.base_fingerprint,
                        spatial_checkpoint_hash=self.spatial_checkpoint_hash,
                        position_offset=self.position_offset, stats=self.stats)
        (directory / "memory.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_memory(self, directory: str | Path):
        from safetensors.torch import load_file

        directory = Path(directory)
        meta = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
        expected_config = json.loads(json.dumps(asdict(self.config.spatial)))
        if (meta.get("format_version") != 1 or meta.get("kind") != "spatial_fast_weights"
                or meta.get("base_fingerprint") != self.base_fingerprint
                or meta.get("spatial_checkpoint_hash") != self.spatial_checkpoint_hash
                or meta["config"]["spatial"] != expected_config):
            raise ValueError("memory/base/spatial configuration mismatch")
        offset = meta.get("position_offset")
        stats = meta.get("stats")
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid position offset")
        if not isinstance(stats, dict) or stats.keys() != self.stats.keys():
            raise ValueError("invalid memory statistics")
        for key, value in stats.items():
            expected_type = (int, float) if key == "ingest_seconds" else (int,)
            if type(value) not in expected_type or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid statistic: {key}")
        state = load_file(str(directory / "fast_weights.safetensors"))
        slow = load_file(str(directory / "spatial.safetensors"))
        if any("pending_" in key for key in state):
            raise ValueError("saved memory must not contain pending visual tokens")
        if bool(stats["chunks"]) != bool(state):
            raise ValueError("memory statistics do not match the saved parameter state")
        # Validate episode tensors before changing slow parameters or existing memory.
        self.controller.validate_memory_state_dict(state)
        self.controller.load_slow_state_dict(slow)
        self.controller.load_memory_state_dict(state)
        self.position_offset, self.stats = offset, stats
        self.state = "ready"
        self._clear_generation_state()


def _config_from_saved(args):
    metadata = json.loads((Path(args.memory) / "memory.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    config.update(device=args.device, dtype=args.dtype)
    if args.model_path:
        config["model_path"] = args.model_path
    if args.spatial_checkpoint:
        config["spatial_checkpoint"] = args.spatial_checkpoint
    return SpatialVideoConfig(**config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_teacher = sub.add_parser(
        "export-teacher", help="sample frames for manual ChatGPT subscription analysis"
    )
    export_teacher.add_argument("--video", required=True)
    export_teacher.add_argument("--out", required=True)
    export_teacher.add_argument("--chunk-seconds", type=float, default=4.0)
    export_teacher.add_argument("--frames-per-chunk", type=int, default=8)
    export_teacher.add_argument("--max-side", type=int, default=448)
    export_teacher.add_argument("--max-chunks", type=int, default=0)
    export_teacher.add_argument("--jpeg-quality", type=int, default=95)
    import_teacher = sub.add_parser(
        "import-teacher", help="validate manual ChatGPT JSON and write canonical analysis"
    )
    import_teacher.add_argument("--packet", required=True,
                               help="packet directory or its manifest.json")
    import_teacher.add_argument("--analysis", required=True,
                               help="JSON copied from the ChatGPT conversation")
    import_teacher.add_argument("--save", required=True,
                               help="output canonical JSON path")
    write = sub.add_parser("ingest", help="direct visual-token writes; no teacher")
    write.add_argument("--model-path", required=True)
    write.add_argument("--video", nargs="+", required=True)
    write.add_argument("--save", required=True)
    write.add_argument("--chunk-seconds", type=float, default=4.0)
    write.add_argument("--frames-per-chunk", type=int, default=8)
    write.add_argument("--max-side", type=int, default=448)
    write.add_argument("--max-chunks", type=int, default=0)
    write.add_argument("--chunk-size", type=int, default=2648)
    write.add_argument("--window-size", type=int, default=2648)
    write.add_argument("--num-heads", type=int, default=4)
    write.add_argument("--base-lr", type=float, default=1e-3)
    write.add_argument("--ttt-scale-init", type=float, default=0.0,
                       help="initial scale bias; nonzero is a diagnostic, not trained memory")
    write.add_argument("--seed", type=int, default=0)
    write.add_argument("--allow-download", action="store_true")
    read = sub.add_parser("ask", help="read saved parameters without video")
    read.add_argument("--memory", required=True)
    read.add_argument("--question", required=True)
    read.add_argument("--without-memory", action="store_true")
    read.add_argument("--max-new-tokens", type=int, default=128)
    resume = sub.add_parser("resume", help="append video to saved memory without resetting it")
    resume.add_argument("--memory", required=True)
    resume.add_argument("--video", nargs="+", required=True)
    resume.add_argument("--save", required=True)
    for command in (write, read, resume):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
        command.add_argument("--spatial-checkpoint", help="full trained official model.safetensors")
        if command is not write:
            command.add_argument("--model-path", help="relocated path to the identical base model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "export-teacher":
        manifest = prepare_teacher_packet(
            args.video, args.out, chunk_seconds=args.chunk_seconds,
            frames_per_chunk=args.frames_per_chunk, max_side=args.max_side,
            max_chunks=args.max_chunks, jpeg_quality=args.jpeg_quality,
        )
        print(json.dumps({"packet": str(Path(args.out)),
                          "manifest_sha256": manifest["manifest_sha256"],
                          "chunks": len(manifest["chunks"]),
                          "frames": sum(len(c["frames"]) for c in manifest["chunks"])},
                         ensure_ascii=False))
        return 0
    if args.command == "import-teacher":
        packet = Path(args.packet)
        manifest_path = packet / "manifest.json" if packet.is_dir() else packet
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis = validate_teacher_analysis(args.analysis, manifest)
        canonical = {"schema": analysis["schema"],
                     "packet_manifest_sha256": analysis["packet_manifest_sha256"],
                     "source_sha256": analysis["source_sha256"],
                     "analysis": analysis,
                     "canonical_text": canonicalize_analysis(analysis)}
        destination = Path(args.save)
        if destination.exists():
            raise FileExistsError(f"output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(json.dumps({"saved": str(destination), "segments": len(analysis["segments"]),
                          "canonical_chars": len(canonical["canonical_text"])},
                         ensure_ascii=False))
        return 0
    if args.command == "ingest":
        spatial = SpatialModelConfig(num_heads=args.num_heads, chunk_size=args.chunk_size,
                                     window_size=args.window_size, base_lr=args.base_lr,
                                     ttt_scale_init=args.ttt_scale_init, seed=args.seed)
        config = SpatialVideoConfig(model_path=args.model_path, device=args.device, dtype=args.dtype,
                                    spatial_checkpoint=args.spatial_checkpoint,
                                    local_files_only=not args.allow_download, spatial=spatial,
                                    chunk_seconds=args.chunk_seconds, frames_per_chunk=args.frames_per_chunk,
                                    max_side=args.max_side, max_chunks=args.max_chunks)
        memory = SpatialVideoMemory(config)
    else:
        memory = SpatialVideoMemory(_config_from_saved(args))
        memory.load_memory(args.memory)
    if args.command == "ask":
        print(memory.answer(args.question, use_memory=not args.without_memory,
                            max_new_tokens=args.max_new_tokens))
    else:
        for video in args.video:
            print(json.dumps(memory.ingest_video(video)), flush=True)
        memory.save(args.save)
        print(json.dumps(memory.finish_ingest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
