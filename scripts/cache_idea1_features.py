#!/usr/bin/env python3
"""Shardable, resumable full-prefix visual features for the Idea 01 evaluation.

Only the public input manifest and its context.mp4/input.mp4 files are opened.
The original extractor is reused without editing it. Explicit instance-local
linear patch embedding and independent-image SDPA preserve attention boundaries.
Different floating-point kernels can round differently: their protocols, source
hashes and batch size are part of the cache identity.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import torch

from ttt_frame.forecast_bench import ForecastCase, load_forecast_cases
from ttt_frame.forecast_features import FeatureConfig, QwenImageFeatures, cache_case
from ttt_frame.vision_attention import EqualLengthVisionSDPA
from ttt_frame.vision_patch_embed import LinearVisionPatchEmbed


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "ttt_frame/forecast_bench.py", "ttt_frame/forecast_features.py",
    "ttt_frame/vision_patch_embed.py", "ttt_frame/vision_attention.py",
    "scripts/cache_idea1_features.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def feature_sha256(features: torch.Tensor) -> str:
    return hashlib.sha256(features.detach().cpu().contiguous().numpy()).hexdigest()


def select_shard(cases: list[ForecastCase], index: int, count: int) -> list[ForecastCase]:
    if isinstance(index, bool) or isinstance(count, bool) or count < 1 or not 0 <= index < count:
        raise ValueError("shard index must satisfy 0 <= index < positive num-shards")
    return [case for ordinal, case in enumerate(cases) if ordinal % count == index]


def media_snapshot(case: ForecastCase) -> dict:
    result = {}
    for role, path in (("context", case.context_path), ("recent", case.recent_path)):
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"source media changed while hashing {path}")
        result[role] = {
            "path": str(path), "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest,
        }
    return result


def build_protocol(config: FeatureConfig, model_fingerprint: dict, manifest_sha: str,
                   batch_size: int) -> dict:
    if config.context_tail_seconds is not None or config.limit_frames:
        raise ValueError("Idea 01 cache requires the complete 1830-second prefix")
    if config.sample_fps != 2:
        raise ValueError("Idea 01 comparison requires exactly 2fps")
    import av
    import PIL
    import transformers

    return {
        "kind": "idea01_full_prefix_visual_cache_protocol", "format_version": 1,
        "feature_config": asdict(config),
        "model_fingerprint": model_fingerprint,
        "input_manifest_sha256": manifest_sha,
        "execution": {
            "linear_vision_patch_embed": LinearVisionPatchEmbed.protocol_name,
            "equal_length_vision_sdpa": EqualLengthVisionSDPA.protocol_name,
            "batch_size": batch_size,
            "images_attend_independently": True,
        },
        "source_sha256": {name: sha256_file(REPO_ROOT / name) for name in SOURCE_FILES},
        "libraries": {
            "torch": str(torch.__version__), "transformers": transformers.__version__,
            "pyav": av.__version__, "pillow": PIL.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "storage_dtype": "float32", "audio_used": False,
        "future_media_used": False, "questions_or_choices_encoded": False,
    }


@contextmanager
def exclusive_directory_lock(path: Path, *, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ValueError(f"active/stale lock {path}; do not launch overlapping workers")
            time.sleep(0.1)
    try:
        yield
    finally:
        path.rmdir()


def install_protocol(output: Path, protocol: dict) -> str:
    output.mkdir(parents=True, exist_ok=True)
    digest = canonical_sha256(protocol)
    record = {"protocol_sha256": digest, "protocol": protocol}
    path = output / "cache_protocol.json"
    with exclusive_directory_lock(output / ".protocol.lock"):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != json.loads(json.dumps(record)):
                raise ValueError(f"conflicting cache protocol in {path}; use another output directory")
        else:
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return digest


def validate_existing(cache: dict, case: ForecastCase, *, protocol_sha: str,
                      protocol: dict, sources: dict) -> None:
    from ttt_frame.forecast_features import sampling_plan

    config = FeatureConfig(**{
        **protocol["feature_config"],
        "pool_shape": tuple(protocol["feature_config"]["pool_shape"]),
    })
    plan = sampling_plan(case, config)
    expected = torch.tensor([sample.prefix_seconds for sample in plan], dtype=torch.float64)
    for key, value in {
        "kind": "bench_v0_causal_visual_features", "format_version": 1,
        "sample_id": case.id, "complete_prefix": True, "cropped": False,
        "protocol_sha256": protocol_sha,
        "input_manifest_sha256": protocol["input_manifest_sha256"],
        "model_fingerprint": protocol["model_fingerprint"],
        "source_media": sources, "audio_used": False,
        "execution": protocol["execution"],
    }.items():
        if cache.get(key) != value:
            raise ValueError(f"conflicting/incomplete cache for {case.id}: {key}")
    # A stale protocol digest must not conceal altered per-cache feature settings.
    if json.loads(json.dumps(cache.get("config"))) != json.loads(json.dumps(asdict(config))):
        raise ValueError(f"conflicting cache for {case.id}: config")
    features = cache.get("features")
    if (not isinstance(features, torch.Tensor) or features.shape != (len(plan), 8, 256)
            or features.dtype != torch.float32 or not bool(torch.isfinite(features).all())):
        raise ValueError(f"invalid feature tensor for {case.id}")
    if cache.get("feature_sha256") != feature_sha256(features):
        raise ValueError(f"corrupt feature checksum for {case.id}")
    requested = cache.get("requested_prefix_seconds")
    actual = cache.get("actual_prefix_seconds")
    if (not isinstance(requested, torch.Tensor) or requested.dtype != torch.float64
            or not torch.equal(requested.cpu(), expected)):
        raise ValueError(f"invalid requested time grid for {case.id}")
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != torch.float64 or not bool(torch.isfinite(actual).all())
            or not bool((actual >= 0).all()) or not bool((actual <= expected + 1e-9).all())
            or not bool((actual[1:] >= actual[:-1]).all())
            or not bool((expected - actual <= 0.5 + 1e-6).all())):
        raise ValueError(f"invalid/possibly noncausal actual time grid for {case.id}")


def atomic_save(cache: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(cache, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/data/quzitsix/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-side", type=int, default=224)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.output_dir.resolve().is_relative_to(args.root.resolve()):
        parser.error("output directory must be outside source benchmark")
    torch.set_num_threads(args.threads)
    config = FeatureConfig(max_side=args.max_side, dtype=args.dtype,
                           projection_seed=args.projection_seed)
    cases = select_shard(load_forecast_cases(args.root, ids=args.ids),
                         args.shard_index, args.num_shards)
    if not cases:
        parser.error("shard has no cases")
    if not args.resume:
        for case in cases:
            if (args.output_dir / f"{case.id}.pt").exists():
                parser.error(f"cache exists for {case.id}; pass --resume to validate and reuse")
    manifest_path = args.root / "inputs_mcq.jsonl"
    manifest_sha = sha256_file(manifest_path)
    print(f"Loading visual tower for shard {args.shard_index}/{args.num_shards}: "
          f"{len(cases)} full-prefix cases on {args.device}", flush=True)
    encoder = QwenImageFeatures(args.model_path, config, args.device)
    protocol = build_protocol(config, encoder.model_fingerprint, manifest_sha, args.batch_size)
    protocol_sha = install_protocol(args.output_dir, protocol)
    print(f"Protocol SHA256: {protocol_sha}", flush=True)
    with LinearVisionPatchEmbed(encoder.model), EqualLengthVisionSDPA(encoder.model):
        for ordinal, case in enumerate(cases, 1):
            destination = args.output_dir / f"{case.id}.pt"
            with exclusive_directory_lock(args.output_dir / f".{case.id}.lock", timeout=1):
                sources = media_snapshot(case)
                if destination.exists():
                    if not args.resume:
                        raise ValueError(f"cache appeared for {case.id}; refusing overwrite")
                    validate_existing(torch.load(destination, map_location="cpu", weights_only=True),
                                      case, protocol_sha=protocol_sha, protocol=protocol,
                                      sources=sources)
                    print(f"[{ordinal}/{len(cases)}] Reused validated {case.id}", flush=True)
                    continue
                start = time.perf_counter()
                last_progress = 0

                def progress(done: int, total: int) -> None:
                    nonlocal last_progress
                    if done - last_progress >= 1024 or done == total:
                        print(f"  {case.id}: {done}/{total} frames", flush=True)
                        last_progress = done

                result = cache_case(case, encoder, input_manifest_sha256=manifest_sha,
                                    batch_size=args.batch_size, progress=progress)
                for role, path in (("context", case.context_path), ("recent", case.recent_path)):
                    current = path.stat()
                    if (current.st_size, current.st_mtime_ns) != (
                        sources[role]["size_bytes"], sources[role]["mtime_ns"]
                    ):
                        raise ValueError(f"source media changed during extraction: {path}")
                if sha256_file(manifest_path) != manifest_sha:
                    raise ValueError("public input manifest changed during extraction")
                result.update(
                    source_media=sources, protocol_sha256=protocol_sha,
                    extraction_seconds=time.perf_counter() - start,
                    execution=protocol["execution"],
                    extraction_device=str(encoder.device),
                    encoder_checkpoint_path=str(args.model_path.resolve()),
                    feature_sha256=feature_sha256(result["features"]),
                )
                validate_existing(result, case, protocol_sha=protocol_sha, protocol=protocol,
                                  sources=sources)
                atomic_save(result, destination)
                print(f"[{ordinal}/{len(cases)}] Saved {destination}: "
                      f"{len(result['features'])} frames, "
                      f"{result['extraction_seconds']:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
