#!/usr/bin/env python3
"""Replay causal 90-frame prefixes and probe native official Spatial fast weights.

This stage never opens answer keys or future media. It writes candidate scores,
not selected hyperparameters. Evaluation and final scoring are separate stages.
The native text-V target is a diagnostic surrogate, not a trained visual mapper.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.evaluate_forecast_mcq import file_hash, fingerprint, read_protocol
from scripts.run_forecast_residual import read_split
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_mcq import load_frame_bundle, write_chunks


SEARCHES = {"search_r01": 0.1, "search_r05": 0.5, "search_r10": 1.0}


def donor_mapping(split: dict) -> dict[str, str]:
    """A fixed train prefix from a different recording day, chosen without labels.

    Val/test donors are strictly earlier train days. Train-day-one diagnostics
    use day-two offline training material as a counterfactual, never own future.
    """
    train = sorted(split["splits"]["train"])
    result = {}
    for role in ("train", "validation", "test"):
        for sample_id in split["splits"][role]:
            eligible = [donor for donor in train
                        if split["id_to_day"][donor] != split["id_to_day"][sample_id]]
            if not eligible:
                raise ValueError("wrong-memory control requires a different train recording day")
            result[sample_id] = eligible[0]
    return result


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict:
    return {name: file_hash(REPO / name) for name in (
        "scripts/run_idea3_spatial.py", "ttt_frame/idea3_spatial_reader.py",
        "ttt_frame/spatial_probe.py", "scripts/run_forecast_residual.py",
    )}


def prepare(args) -> dict:
    if args.protocol.exists():
        raise FileExistsError("protocol exists; use a new path")
    baseline = read_protocol(args.baseline_protocol)
    split = read_split(args.split_manifest)
    p = baseline["payload"]
    if args.protocol.resolve().is_relative_to(Path(p["root"]).resolve()):
        raise ValueError("protocol must be outside source benchmark")
    ids = set().union(*(set(values) for values in split["splits"].values()))
    if ids != set(p["case_ids"]) or split["inputs_manifest_sha256"] != p["inputs_sha256"]:
        raise ValueError("baseline/split/public inputs disagree")
    payload = {
        "schema": "idea03_native_spatial_probe_v1", "baseline_protocol": baseline,
        "baseline_protocol_sha256": file_hash(args.baseline_protocol),
        "split": split, "split_sha256": file_hash(args.split_manifest),
        "sources": source_hashes(), "donors": donor_mapping(split),
        "searches": SEARCHES, "steps": 3, "step_size": 1.0, "regularization": 0.01,
        "seed": 0, "target": "per-head unit-normalized native text V at final option token",
        "address": "native question-only final prompt token Q under reset state",
        "score": "head/layer mean episode-minus-reset association, minus address penalty",
        "prior": "reset native conditional candidate mean token log likelihood",
        "training": "none; validation-only scalar/radius selection in separate stage",
        "write_protocol": "unchanged baseline 21-layer native Spatial, discard cross-clip KV",
        "not_official_full_streaming_protocol": True,
        "candidate_letters_or_positions_encoded": False,
        "future_media_used": False, "answer_keys_used": False,
        "wrong_memory_policy": "first sorted train ID of a different day; recipient query coordinates retained",
    }
    record = {"payload": payload, "protocol_fingerprint": fingerprint(payload),
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "created_at": datetime.now(timezone.utc).isoformat()}
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.protocol, record)
    return record


def validate_protocol(path: Path) -> dict:
    record = json.loads(path.read_text())
    p = record["payload"]
    if fingerprint(p) != record["protocol_fingerprint"] or p["sources"] != source_hashes():
        raise ValueError("Idea3 protocol or implementation changed")
    baseline = p["baseline_protocol"]
    from scripts.evaluate_forecast_mcq import source_hashes as baseline_sources
    if (fingerprint(baseline["payload"]) != baseline["protocol_fingerprint"]
            or baseline["payload"]["source_hashes"] != baseline_sources()):
        raise ValueError("baseline source protocol changed")
    root = Path(baseline["payload"]["root"])
    if file_hash(root / "inputs_mcq.jsonl") != baseline["payload"]["inputs_sha256"]:
        raise ValueError("public inputs changed")
    if p["donors"] != donor_mapping(p["split"]):
        raise ValueError("donor mapping changed")
    return record


def completed(output: Path, selected: set, digest: str) -> set:
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if (row["id"] not in selected or row["id"] in done
                    or row.get("protocol_fingerprint") != digest or row.get("status") != "ok"):
                raise ValueError("invalid, duplicate, foreign, or mixed-protocol completed row")
            if (row.get("schema") != "idea03_native_spatial_scores_v1"
                    or row.get("answer_keys_used") is not False
                    or len(row.get("labels", [])) != 6 or set(row["labels"]) != set("ABCDEF")
                    or len(row.get("option_texts", [])) != 6
                    or any(not isinstance(value, str) or not value.strip() for value in row["option_texts"])):
                raise ValueError("incomplete completed score row")
            expected = {"direct", "wrong_direct", "reset_direct", "reset_search"} | {
                prefix + search for prefix in ("", "wrong_") for search in SEARCHES}
            if set(row.get("scores", {})) != expected:
                raise ValueError("incomplete completed score arms")
            for values in [row.get("prior_scores", []), *row["scores"].values()]:
                if len(values) != 6 or any(type(value) not in (int, float) or not math.isfinite(value)
                                           for value in values):
                    raise ValueError("completed scores must contain six finite numbers")
            done.add(row["id"])
    return done


def state_to(states, device):
    return {idx: replace(state, **{name: getattr(state, name).to(device) for name in (
        "w0", "w1", "w2", "w0_norm", "w1_norm", "w2_norm")}) for idx, state in states.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--baseline-protocol", type=Path,
                        default=Path("runs/forecast_bench_v0/mcq_baseline_v1/protocol.json"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("runs/forecast_bench_v0/research_day_split.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid threads/shard")
    if args.prepare:
        record = prepare(args)
        print(json.dumps({"protocol": str(args.protocol), "fingerprint": record["protocol_fingerprint"]}), flush=True)
        return 0
    record = validate_protocol(args.protocol)
    p = record["payload"]
    base = p["baseline_protocol"]["payload"]
    root, sampling = Path(base["root"]), base["sampling"]
    if args.output is None or args.output.resolve().is_relative_to(root.resolve()):
        parser.error("--output outside source benchmark required")
    if args.output.resolve() == args.protocol.resolve():
        parser.error("output cannot replace protocol")
    if args.output.exists() and not args.resume:
        parser.error("output exists; explicit --resume required")
    all_cases = load_forecast_cases(root)
    by_id = {case.id: case for case in all_cases}
    if args.ids and (len(set(args.ids)) != len(args.ids) or not set(args.ids) <= set(by_id)):
        parser.error("invalid --ids")
    cases = [case for case in all_cases if not args.ids or case.id in args.ids]
    cases = [case for i, case in enumerate(cases) if i % args.num_shards == args.shard_index]
    done = completed(args.output, {case.id for case in cases}, record["protocol_fingerprint"])
    pending = [case for case in cases if case.id not in done]
    if not pending:
        print("All selected cases already completed.", flush=True)
        return 0
    import torch
    from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory
    from ttt_frame.idea3_spatial_reader import (
        NativeQKVCapture, extract_text_addresses, score_addresses,
        snapshot_states, visual_key_diagnostic,
    )

    torch.set_num_threads(args.threads)
    torch.manual_seed(p["seed"])
    for name, expected in base["base_artifact_sha256"].items():
        if file_hash(Path(base["model_path"]) / name) != expected:
            raise ValueError(f"base model artifact changed: {name}")
    if file_hash(Path(base["spatial_checkpoint"])) != base["spatial_checkpoint_sha256"]:
        raise ValueError("official checkpoint changed")
    engine = SpatialVideoMemory(SpatialVideoConfig(
        model_path=base["model_path"], device=args.device, dtype=base["dtype"],
        spatial_checkpoint=base["spatial_checkpoint"], max_side=sampling["max_side"]))
    engine.model.requires_grad_(False)
    if base["vision_linear_patch"]:
        from ttt_frame.vision_patch_embed import LinearVisionPatchEmbed
        engine.vision_patch_embedding = LinearVisionPatchEmbed(engine.model)
    if base["vision_batched_sdpa"]:
        from ttt_frame.vision_attention import EqualLengthVisionSDPA
        engine.vision_attention_patch = EqualLengthVisionSDPA(engine.model)
    reset = snapshot_states(engine, reset=True)

    def ingest(case):
        begin = time.perf_counter()
        bundle = load_frame_bundle(case, context_frames=sampling["context_frames"],
                                   recent_frames=sampling["recent_frames"], max_side=sampling["max_side"])
        decode_seconds = time.perf_counter() - begin
        engine.reset()
        with NativeQKVCapture(engine, selection="last_visual") as capture:
            for role in ("context", "recent"):
                pairs = list(zip(bundle[f"{role}_images"], bundle[f"{role}_actual_seconds"]))
                for chunk in write_chunks(pairs, sampling["write_chunk_frames"]):
                    images, timestamps = zip(*chunk)
                    engine.ingest_frames(list(images), list(timestamps))
        report = engine.finish_ingest()
        audit = {key: value for key, value in bundle.items() if not key.endswith("_images")}
        return report, audit, decode_seconds, capture.records

    donor_states, donor_audits = {}, {}
    for donor in sorted({p["donors"][case.id] for case in pending}):
        report, audit, decode_seconds, _ = ingest(by_id[donor])
        donor_states[donor] = snapshot_states(engine, device="cpu")
        donor_audits[donor] = {"write": report, "sampling": audit, "decode_seconds": decode_seconds}
        print(json.dumps({"donor_ready": donor, "frames": report["stats"]["frames"], "device": args.device}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    donor_path = args.output.with_suffix(".donors.json")
    if donor_path.exists():
        previous = json.loads(donor_path.read_text())
        for donor, audit in donor_audits.items():
            if donor in previous and previous[donor]["sampling"] != audit["sampling"]:
                raise ValueError("donor media changed on resume")
        donor_audits = {**previous, **donor_audits}
    atomic_json(donor_path, donor_audits)
    start = time.perf_counter()
    with args.output.open("a" if args.resume else "x") as destination:
        for count, case in enumerate(pending, 1):
            begin = time.perf_counter()
            report, audit, decode_seconds, visual_pairs = ingest(case)
            current = snapshot_states(engine)
            query_start = time.perf_counter()
            addresses = extract_text_addresses(engine, case.question, [option.text for option in case.options])
            extraction_seconds = time.perf_counter() - query_start
            donor = p["donors"][case.id]
            wrong = state_to(donor_states[donor], args.device)
            visual_start = time.perf_counter()
            visual_diagnostic = visual_key_diagnostic(current, reset, addresses, visual_pairs)
            visual_seconds = time.perf_counter() - visual_start
            scores, diagnostics, timings = {}, {}, {}
            for name, state in (("", current), ("wrong_", wrong)):
                for search_name, radius in p["searches"].items():
                    score_start = time.perf_counter()
                    result = score_addresses(state, reset, addresses, radius=radius, steps=p["steps"],
                                             step_size=p["step_size"], regularization=p["regularization"])
                    scores[name + "direct"] = result["direct_scores"]
                    scores[name + search_name] = result["search_objectives"]
                    timings[name + search_name] = time.perf_counter() - score_start
                    diagnostics[name + search_name] = result
            reset_start = time.perf_counter()
            zero = score_addresses(reset, reset, addresses, radius=1.0, steps=3,
                                   step_size=p["step_size"], regularization=p["regularization"])
            reset_seconds = time.perf_counter() - reset_start
            scores["reset_direct"], scores["reset_search"] = zero["direct_scores"], zero["search_objectives"]
            if any(value != 0.0 for values in (scores["reset_direct"], scores["reset_search"]) for value in values):
                raise RuntimeError("empty memory association must be exactly zero")
            row = {
                "schema": "idea03_native_spatial_scores_v1", "id": case.id, "status": "ok",
                "research_split": next(role for role, ids in p["split"]["splits"].items() if case.id in ids),
                "labels": [option.label for option in case.options],
                "option_texts": [option.text for option in case.options],
                "prior_scores": addresses.prior_scores, "scores": scores, "diagnostics": diagnostics,
                "visual_key_diagnostic": visual_diagnostic,
                "protocol_fingerprint": record["protocol_fingerprint"], "donor_id": donor,
                "sampling": audit, "write_report": report, "device": args.device,
                "query_tokens": addresses.query_tokens, "candidate_tokens": addresses.candidate_tokens,
                "position_offset": engine.position_offset, "decode_seconds": decode_seconds,
                "extraction_seconds": extraction_seconds, "search_seconds": timings,
                "visual_probe_seconds": visual_seconds, "reset_probe_seconds": reset_seconds,
                "elapsed_seconds": time.perf_counter() - begin,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
                "torch_version": str(torch.__version__), "answer_keys_used": False,
            }
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            destination.flush()
            del current, wrong, addresses
            print(json.dumps({"completed": count, "total": len(pending), "id": case.id,
                              "seconds": row["elapsed_seconds"], "query_seconds": extraction_seconds,
                              "search_seconds": sum(timings.values())}), flush=True)
    validate_protocol(args.protocol)
    atomic_json(args.output.with_suffix(".completion.json"), {
        "complete": True, "rows": len(cases), "scores_sha256": file_hash(args.output),
        "protocol_fingerprint": record["protocol_fingerprint"], "test_answers_loaded": False,
        "elapsed_seconds": time.perf_counter() - start, "reused_cases": len(done),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
