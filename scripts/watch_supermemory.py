#!/usr/bin/env python3
"""Inspect and play Supermemory clips in chronological order.

The benchmark stores each long recording as 60-second clip files and keeps the
ordering in ``media_index.json``.  This utility deliberately does not decode
or train a model: it resolves that manifest and delegates playback/concatenation
to ffplay/ffmpeg.

Examples
--------
List the 18 clips in the 1080-second Q9 recording::

    python scripts/watch_supermemory.py list

Play them one after another (close one ffplay window to advance)::

    python scripts/watch_supermemory.py play --fullscreen

Print one ordered path per line for a model-ingestion command::

    python scripts/watch_supermemory.py paths > /tmp/q9-ordered.txt

Create one chronological file for seeking in a normal video player::

    python scripts/watch_supermemory.py concat --output /tmp/q9-full.mp4

Paths can be overridden for another checkout or another downloaded media set
with ``--media-index``, ``--memory-id`` and ``--media-root``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MEDIA_INDEX = Path(
    "/data/quzitsix/meow-releases/supermemory-pilot-v2/media_index.json"
)
DEFAULT_MEMORY_ID = "sm-7a3e7974f1304cc9c19b"
DEFAULT_MEDIA_ROOT = Path("/data/quzitsix/meow-releases/supermemory-pilot-v2/media")


@dataclass(frozen=True)
class Clip:
    """One manifest entry and the media path resolved from its session id."""

    ordinal: int
    session_id: str
    start_sec: float
    end_sec: float
    path: Path

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _as_float(value: Any, field: str, index: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest entry {index} has invalid {field}: {value!r}") from exc


def load_clips(
    media_index: Path,
    *,
    memory_id: str,
    media_root: Path,
    allow_missing: bool = False,
) -> list[Clip]:
    """Load and chronologically sort one recording from ``media_index.json``.

    The benchmark has used both a dictionary keyed by memory id and a bare list
    in small fixtures, so both forms are accepted.  A clip is resolved by its
    ``session_id`` (for example ``clip-eaf...`` -> ``clip-eaf....mp4``).
    """

    if not media_index.is_file():
        raise FileNotFoundError(f"media index does not exist: {media_index}")
    with media_index.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if memory_id not in payload:
            available = ", ".join(str(key) for key in payload.keys())
            raise KeyError(f"memory id {memory_id!r} not found; available ids: {available}")
        entries = payload[memory_id]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("media index must contain a list or an object keyed by memory id")
    if not isinstance(entries, list):
        raise ValueError(f"manifest value for {memory_id!r} is not a list")

    clips: list[Clip] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"manifest entry {index} is not an object")
        session_id = str(item.get("session_id", "")).strip()
        if not session_id:
            raise ValueError(f"manifest entry {index} has no session_id")
        start_sec = _as_float(item.get("start_sec"), "start_sec", index)
        end_sec = _as_float(item.get("end_sec"), "end_sec", index)
        if end_sec < start_sec:
            raise ValueError(f"manifest entry {index} ends before it starts")

        # Permit an absolute path in hand-written manifests, while retaining
        # the benchmark's normal session-id -> <root>/<id>.mp4 convention.
        candidate = Path(session_id)
        if not candidate.is_absolute():
            candidate = media_root / session_id
        if candidate.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm"}:
            candidate = candidate.with_suffix(".mp4")
        if not candidate.is_file() and not allow_missing:
            raise FileNotFoundError(
                f"media for manifest entry {index} ({session_id}) not found: {candidate}"
            )
        clips.append(Clip(index + 1, session_id, start_sec, end_sec, candidate))

    # The JSON is usually already ordered, but sorting makes the guarantee
    # explicit and protects against a shuffled export.
    clips.sort(key=lambda clip: (clip.start_sec, clip.end_sec, clip.session_id))
    return [
        Clip(index + 1, clip.session_id, clip.start_sec, clip.end_sec, clip.path)
        for index, clip in enumerate(clips)
    ]


def select_clips(clips: Sequence[Clip], start: int = 1, end: int | None = None) -> list[Clip]:
    """Select one-based inclusive clip ordinals."""

    if start < 1:
        raise ValueError("--start must be at least 1")
    if end is not None and end < start:
        raise ValueError("--end must be greater than or equal to --start")
    selected = list(clips[start - 1 : end])
    if not selected:
        raise ValueError(f"clip range {start}:{end or 'end'} is empty (there are {len(clips)})")
    return selected


def format_timestamp(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_manifest(clips: Iterable[Clip], *, media_index: Path, memory_id: str) -> None:
    clips = list(clips)
    total = sum(clip.duration for clip in clips)
    print(f"media_index: {media_index}")
    print(f"memory_id:   {memory_id}")
    print(f"clips:       {len(clips)}   duration: {format_timestamp(total)} ({total:.1f}s)")
    print()
    print(" #   time range             duration  status  file")
    print("---  ---------------------  --------  ------  ----------------------------------------")
    for index, clip in enumerate(clips, 1):
        status = "ok" if clip.path.is_file() else "MISSING"
        name = str(clip.path)
        print(
            f"{index:2d}   {format_timestamp(clip.start_sec)} - "
            f"{format_timestamp(clip.end_sec)}  {clip.duration:7.1f}s  {status:7s}  {name}"
        )


def print_paths(clips: Iterable[Clip]) -> None:
    """Emit only resolved media paths, suitable for shell command substitution."""

    for clip in clips:
        print(os.fspath(clip.path))


def require_program(program: str) -> None:
    if shutil.which(program) is None:
        raise RuntimeError(
            f"{program!r} was not found on PATH; install ffmpeg (which provides ffplay) "
            "or pass --player to a compatible player"
        )


def play_clips(
    clips: Sequence[Clip],
    *,
    player: str,
    fullscreen: bool,
    mute: bool,
    extra_args: Sequence[str],
) -> None:
    """Run one player process per clip, preserving chronological order."""

    require_program(player)
    for index, clip in enumerate(clips, 1):
        if not clip.path.is_file():
            raise FileNotFoundError(f"cannot play missing clip {clip.path}")
        print(
            f"[{index}/{len(clips)}] {format_timestamp(clip.start_sec)}--"
            f"{format_timestamp(clip.end_sec)}  {clip.path}",
            flush=True,
        )
        command = [player, "-autoexit", "-loglevel", "warning"]
        if fullscreen and Path(player).name == "ffplay":
            command.append("-fs")
        if mute and Path(player).name == "ffplay":
            command.append("-an")
        if Path(player).name == "ffplay":
            command.extend(["-window_title", f"Supermemory {index}/{len(clips)}"])
        command.extend(extra_args)
        command.append(os.fspath(clip.path))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"player exited with status {result.returncode}: {shlex.join(command)}")


def _concat_manifest(clips: Sequence[Clip], handle: Any) -> None:
    for clip in clips:
        if not clip.path.is_file():
            raise FileNotFoundError(f"cannot concatenate missing clip {clip.path}")
        # ffmpeg's concat demuxer uses single-quoted paths.  Escape embedded
        # quotes in the standard way; absolute paths are used with -safe 0.
        escaped = os.fspath(clip.path).replace("'", "'\\''")
        handle.write(f"file '{escaped}'\n")


def concat_clips(
    clips: Sequence[Clip], *, output: Path, ffmpeg: str, reencode: bool, overwrite: bool
) -> None:
    require_program(ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        manifest_path = Path(handle.name)
        _concat_manifest(clips, handle)
    try:
        command = [ffmpeg]
        if overwrite:
            command.append("-y")
        else:
            command.append("-n")
        command.extend(["-f", "concat", "-safe", "0", "-i", os.fspath(manifest_path)])
        if reencode:
            command.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac"])
        else:
            command.extend(["-c", "copy"])
        command.append(os.fspath(output))
        print(f"Creating {output} from {len(clips)} clips...")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed. The clips may have incompatible codecs; retry with "
                f"--reencode. Command: {shlex.join(command)}"
            )
    finally:
        manifest_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("list", "paths", "play", "concat"))
    parser.add_argument("--media-index", type=Path, default=DEFAULT_MEDIA_INDEX)
    parser.add_argument("--memory-id", default=DEFAULT_MEMORY_ID)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--allow-missing", action="store_true", help="list entries even if a clip file is absent")
    parser.add_argument("--start", type=int, default=1, help="one-based first clip to use (default: 1)")
    parser.add_argument("--end", type=int, help="one-based last clip to use, inclusive")

    # Playback options.
    parser.add_argument("--player", default="ffplay", help="player executable (default: ffplay)")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--player-arg", action="append", default=[], help="extra player argument; repeatable")

    # Concatenation options.
    parser.add_argument("--output", type=Path, help="output video path for the concat command")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--reencode", action="store_true", help="re-encode instead of stream-copying")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        clips = load_clips(
            args.media_index,
            memory_id=args.memory_id,
            media_root=args.media_root,
            allow_missing=args.allow_missing,
        )
        selected = select_clips(clips, args.start, args.end)
        if args.command == "list":
            print_manifest(selected, media_index=args.media_index, memory_id=args.memory_id)
        elif args.command == "paths":
            print_paths(selected)
        elif args.command == "play":
            play_clips(
                selected,
                player=args.player,
                fullscreen=args.fullscreen,
                mute=args.mute,
                extra_args=args.player_arg,
            )
        else:
            if args.output is None:
                raise ValueError("the concat command requires --output PATH")
            concat_clips(
                selected,
                output=args.output,
                ffmpeg=args.ffmpeg,
                reencode=args.reencode,
                overwrite=args.overwrite,
            )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
