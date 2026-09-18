"""Frame packets and validated scene data for manual or Codex CLI teachers.

Codex CLI owns subscription authentication; this module never accesses credentials.
Raw media and teacher text remain outside the parameter-memory checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .video import iter_video_chunks

PACKET_SCHEMA = "ttt_frame.teacher_packet/1"
ANALYSIS_SCHEMA = "ttt_frame.teacher_scene/1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_object(obj: Mapping[str, Any], field: str) -> str:
    body = {k: v for k, v in obj.items() if k != field}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _finite_number(value: Any, *, name: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return number


def _text(value: Any, *, name: str, max_chars: int = 16_384) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise ValueError(f"{name} must be a non-empty string of at most {max_chars} characters")
    return " ".join(value.split())


def _string_list(value: Any, *, name: str, max_items: int = 256) -> list[str]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{name} must be a list with at most {max_items} items")
    return [_text(item, name=f"{name}[{i}]") for i, item in enumerate(value)]


def _manifest_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "manifest_sha256"}


def prepare_teacher_packet(
    video: str | Path,
    out: str | Path,
    *,
    chunk_seconds: float = 4.0,
    frames_per_chunk: int = 8,
    max_side: int = 448,
    max_chunks: int = 0,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """Sample ``video`` and write a manually uploadable teacher packet.

    ``out`` is a new directory containing JPEG frames, ``manifest.json`` and an
    exact prompt contract.  The source video itself is never copied into it.
    """
    source = Path(video)
    if not source.is_file():
        raise FileNotFoundError(source)
    if Path(out).exists():
        raise FileExistsError(f"output already exists: {out}")
    if (not math.isfinite(chunk_seconds) or chunk_seconds <= 0 or
            isinstance(frames_per_chunk, bool) or frames_per_chunk < 1 or
            isinstance(max_side, bool) or max_side < 1 or
            isinstance(max_chunks, bool) or max_chunks < 0 or
            isinstance(jpeg_quality, bool) or not 1 <= jpeg_quality <= 100):
        raise ValueError("invalid teacher packet sampling settings")

    target = Path(out)
    target.mkdir(parents=True)
    frames_dir = target / "frames"
    frames_dir.mkdir()
    chunks: list[dict[str, Any]] = []
    try:
        iterator = iter_video_chunks(source, chunk_seconds=chunk_seconds,
                                     frames_per_chunk=frames_per_chunk,
                                     max_side=max_side, max_chunks=max_chunks)
        with closing(iterator):
            for chunk in iterator:
                frame_entries = []
                chunk_dir = frames_dir / f"chunk-{chunk.index:06d}"
                chunk_dir.mkdir()
                for frame_no, (image, timestamp) in enumerate(zip(chunk.images, chunk.timestamps)):
                    filename = f"frame-{frame_no:04d}-{timestamp:012.3f}.jpg"
                    path = chunk_dir / filename
                    image.save(path, format="JPEG", quality=jpeg_quality, optimize=True)
                    frame_entries.append({
                        "file": path.relative_to(target).as_posix(),
                        "timestamp_sec": round(float(timestamp), 6),
                        "sha256": _sha256_file(path),
                        "width": image.width,
                        "height": image.height,
                    })
                if frame_entries:
                    chunks.append({"chunk_index": int(chunk.index),
                                   "start_sec": frame_entries[0]["timestamp_sec"],
                                   "end_sec": frame_entries[-1]["timestamp_sec"],
                                   "frames": frame_entries})
    except Exception:
        # Keep an interrupted packet from being mistaken for a valid one.
        import shutil
        shutil.rmtree(target, ignore_errors=True)
        raise
    if not chunks:
        import shutil
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError("video yielded no sampled frames")

    manifest: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "source_name": source.name,
        "source_sha256": _sha256_file(source),
        "sampling": {"chunk_seconds": float(chunk_seconds),
                      "frames_per_chunk": int(frames_per_chunk),
                      "max_side": int(max_side), "max_chunks": int(max_chunks)},
        "chunks": chunks,
    }
    manifest["manifest_sha256"] = _digest_object(manifest, "manifest_sha256")
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                       indent=2) + "\n", encoding="utf-8")
    (target / "teacher_prompt.md").write_text(_prompt(manifest), encoding="utf-8")
    return manifest


def _prompt(manifest: Mapping[str, Any]) -> str:
    refs = sum(len(c["frames"]) for c in manifest["chunks"])
    first = manifest["chunks"][0]
    example_ref = first["frames"][0]["file"]
    example_start = first["start_sec"]
    example_end = first["end_sec"]
    return f"""# External scene teacher task

This packet contains {len(manifest['chunks'])} chronological chunks and {refs} sampled frames.
Open every frame and describe only visible evidence. Preserve the supplied timestamps and
frame references; do not infer facts that are not visible. Return **JSON only** (no Markdown
fences) matching `ttt_frame.teacher_scene/1` exactly:

```json
{{
  "schema": "ttt_frame.teacher_scene/1",
  "packet_manifest_sha256": "{manifest['manifest_sha256']}",
  "source_sha256": "{manifest['source_sha256']}",
  "segments": [
    {{
      "chunk_index": 0,
      "start_sec": {example_start},
      "end_sec": {example_end},
      "frame_refs": ["{example_ref}"],
      "summary": "short visible scene summary",
      "observations": ["observable fact"],
      "entities": [{{"id":"object-1","type":"object","name":"cup","attributes":["white"]}}],
      "events": [{{"type":"moved","subject":"object-1","object":"table","location":"kitchen","start_sec":{example_start},"end_sec":{example_end}}}],
      "relations": ["object-1 is on the table"],
      "uncertainty": ["identity of person is unclear"]
    }}
  ],
  "global_summary": "facts visible across the packet"
}}
```

Include one segment per packet chunk where possible. Keep every event interval within its
segment. Use an empty list when a category has no evidence, and use `uncertainty` instead
of guessing. Never include answers to benchmark questions or information from outside the
frames.
"""


def _load_json(path: str | Path) -> dict[str, Any]:
    if Path(path).stat().st_size > 4 * 1024 * 1024:
        raise ValueError("teacher JSON exceeds 4 MiB")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid teacher JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("teacher analysis must be a JSON object")
    return value


def _packet_maps(manifest: Mapping[str, Any]):
    if manifest.get("schema") != PACKET_SCHEMA or manifest.get("manifest_sha256") != _digest_object(manifest, "manifest_sha256"):
        raise ValueError("packet manifest schema/hash mismatch")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("packet manifest sampling is missing")
    _finite_number(sampling.get("chunk_seconds"), name="sampling.chunk_seconds", minimum=1e-12)
    for key in ("frames_per_chunk", "max_side", "max_chunks"):
        value = sampling.get(key)
        invalid = (isinstance(value, bool) or not isinstance(value, int) or
                   (key != "max_chunks" and value < 1) or
                   (key == "max_chunks" and value < 0))
        if invalid:
            raise ValueError(f"invalid sampling.{key}")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("packet manifest has no chunks")
    refs, ranges = {}, {}
    previous = -1.0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("packet chunk must be an object")
        idx = chunk.get("chunk_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx in ranges:
            raise ValueError("invalid or duplicate packet chunk_index")
        start = _finite_number(chunk.get("start_sec"), name="chunk.start_sec")
        end = _finite_number(chunk.get("end_sec"), name="chunk.end_sec")
        if end < start or start < previous:
            raise ValueError("packet chunk times must be ordered")
        previous = end
        ranges[idx] = (start, end)
        frames = chunk.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("packet chunk must contain frames")
        previous_frame = -1.0
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("packet frame must be an object")
            ref = frame.get("file")
            ts = _finite_number(frame.get("timestamp_sec"), name="frame.timestamp_sec")
            posix_ref = PurePosixPath(ref) if isinstance(ref, str) else None
            if (not isinstance(ref, str) or ref in refs or not ref.startswith("frames/")
                    or posix_ref.is_absolute() or ".." in posix_ref.parts):
                raise ValueError("invalid or duplicate frame reference")
            if ts < previous_frame or ts < start - 1e-3 or ts > end + 1e-3:
                raise ValueError("frame timestamp outside its packet chunk")
            previous_frame = ts
            refs[ref] = (ts, idx)
    return refs, ranges


def validate_teacher_analysis(path: Mapping[str, Any] | str | Path,
                              manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate teacher JSON against a packet manifest and return its object."""
    if isinstance(manifest, (str, Path)):
        manifest = _load_json(manifest)
    refs, ranges = _packet_maps(manifest)
    if isinstance(path, Mapping):
        serialized = json.dumps(path, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > 4 * 1024 * 1024:
            raise ValueError("teacher JSON exceeds 4 MiB")
        analysis = json.loads(serialized)
    else:
        analysis = _load_json(path)
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError(f"analysis schema must be {ANALYSIS_SCHEMA}")
    if analysis.get("packet_manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("analysis references a different packet manifest")
    if analysis.get("source_sha256") != manifest.get("source_sha256"):
        raise ValueError("analysis references a different source video")
    segments = analysis.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("analysis.segments must be a non-empty list")
    seen: set[int] = set()
    previous_index = -1
    for n, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{n}] must be an object")
        idx = segment.get("chunk_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx not in ranges or idx in seen:
            raise ValueError(f"segments[{n}] has invalid or duplicate chunk_index")
        seen.add(idx)
        if idx <= previous_index:
            raise ValueError("analysis segments must be chronological")
        previous_index = idx
        start = _finite_number(segment.get("start_sec"), name=f"segments[{n}].start_sec")
        end = _finite_number(segment.get("end_sec"), name=f"segments[{n}].end_sec")
        lo, hi = ranges[idx]
        if end < start or start < lo - 1e-3 or end > hi + 1e-3:
            raise ValueError(f"segments[{n}] time range is outside packet chunk")
        frame_refs = segment.get("frame_refs")
        if not isinstance(frame_refs, list) or not frame_refs:
            raise ValueError(f"segments[{n}].frame_refs must be a non-empty list")
        if (any(not isinstance(ref, str) for ref in frame_refs)
                or len(set(frame_refs)) != len(frame_refs) or any(ref not in refs for ref in frame_refs)):
            raise ValueError(f"segments[{n}] references a frame outside the packet")
        if any(refs[ref][1] != idx for ref in frame_refs):
            raise ValueError(f"segments[{n}] references a frame from another chunk")
        if any(not start - 1e-3 <= refs[ref][0] <= end + 1e-3 for ref in frame_refs):
            raise ValueError(f"segments[{n}] frame timestamp outside segment")
        _text(segment.get("summary"), name=f"segments[{n}].summary")
        _string_list(segment.get("observations", []), name=f"segments[{n}].observations")
        _string_list(segment.get("relations", []), name=f"segments[{n}].relations")
        _string_list(segment.get("uncertainty", []), name=f"segments[{n}].uncertainty")
        qa = segment.get("qa", [])
        if not isinstance(qa, list) or len(qa) > 32:
            raise ValueError("segment.qa must contain at most 32 pairs")
        for pair in qa:
            if not isinstance(pair, dict) or set(pair) != {"question", "answer"}:
                raise ValueError("each QA pair needs question and answer")
            _text(pair["question"], name="qa.question", max_chars=2048)
            _text(pair["answer"], name="qa.answer", max_chars=4096)
        entities = segment.get("entities", [])
        if not isinstance(entities, list) or len(entities) > 256:
            raise ValueError(f"segments[{n}].entities must be a list")
        for j, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(f"entities[{j}] must be an object")
            _text(entity.get("id"), name=f"entities[{j}].id", max_chars=128)
            _text(entity.get("type"), name=f"entities[{j}].type", max_chars=128)
            _text(entity.get("name"), name=f"entities[{j}].name", max_chars=256)
            _string_list(entity.get("attributes", []), name=f"entities[{j}].attributes", max_items=64)
        events = segment.get("events", [])
        if not isinstance(events, list) or len(events) > 256:
            raise ValueError(f"segments[{n}].events must be a list")
        for j, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"events[{j}] must be an object")
            for key in ("type", "subject", "object", "location"):
                _text(event.get(key), name=f"events[{j}].{key}", max_chars=256)
            event_start = _finite_number(event.get("start_sec"), name="event.start_sec")
            event_end = _finite_number(event.get("end_sec"), name="event.end_sec")
            if event_end < event_start or event_start < start - 1e-3 or event_end > end + 1e-3:
                raise ValueError(f"events[{j}] interval outside segment")
    if seen != set(ranges):
        raise ValueError("analysis must cover every packet chunk")
    _text(analysis.get("global_summary"), name="global_summary")
    return analysis


def canonicalize_analysis(analysis: Mapping[str, Any]) -> str:
    """Render a validated analysis to deterministic, bounded teacher text."""
    if not isinstance(analysis, Mapping) or analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("canonicalize_analysis requires validated teacher analysis")
    lines = ["External visual scene analysis (timestamped evidence):"]
    for segment in sorted(analysis["segments"], key=lambda item: item["chunk_index"]):
        lines.append(f"[{segment['start_sec']:.3f}-{segment['end_sec']:.3f}s] {segment['summary']}")
        for item in segment.get("observations", []):
            lines.append(f"Observation: {item}")
        for entity in segment.get("entities", []):
            attrs = ", ".join(entity.get("attributes", []))
            lines.append(f"Entity {entity['id']} ({entity['type']}): {entity['name']}" + (f"; {attrs}" if attrs else ""))
        for event in segment.get("events", []):
            lines.append(f"Event {event['start_sec']:.3f}-{event['end_sec']:.3f}s: {event['type']}; subject={event['subject']}; object={event['object']}; location={event['location']}")
        for item in segment.get("relations", []):
            lines.append(f"Relation: {item}")
        for item in segment.get("uncertainty", []):
            lines.append(f"Uncertain: {item}")
    lines.append(f"Global summary: {analysis['global_summary']}")
    return "\n".join(lines)


__all__ = ["PACKET_SCHEMA", "ANALYSIS_SCHEMA", "prepare_teacher_packet",
           "validate_teacher_analysis", "canonicalize_analysis"]
