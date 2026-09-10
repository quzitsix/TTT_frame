"""Bounded, chronological RGB sampling. No dependency on the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class VideoChunk:
    index: int
    timestamps: list[float]
    images: list[Any]


def iter_video_chunks(
    path: str | Path, *, chunk_seconds: float = 30.0, frames_per_chunk: int = 4,
    max_side: int = 448, max_chunks: int = 0,
) -> Iterator[VideoChunk]:
    """Decode once, without seeks; keep at most one chunk's sampled PIL images.

    Timestamps are seconds from the first decoded frame. Sample the start of
    each equal subinterval, so a short final chunk still gets its first frame.
    max_chunks=0 means the whole video; a nonzero cap is an explicit pilot budget.
    The caller must close the generator if it stops consuming early.
    """
    import av

    if chunk_seconds <= 0 or frames_per_chunk < 1 or max_side < 1 or max_chunks < 0:
        raise ValueError("invalid video sampling budget")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream: {path}")
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else None
        first_time = None
        previous = -1.0
        chunk = None
        yielded = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            absolute = frame.time
            if absolute is None:
                if not rate:
                    raise ValueError("video has neither timestamps nor a usable frame rate")
                absolute = frame_index / rate
            if first_time is None:
                first_time = absolute
            timestamp = max(0.0, float(absolute - first_time))
            if timestamp + 1e-6 < previous:
                raise ValueError("non-monotonic video timestamps")
            previous = timestamp
            index = int(timestamp / chunk_seconds)
            if chunk is None or index != chunk.index:
                if chunk is not None:
                    yield chunk
                    yielded += 1
                    if max_chunks and yielded >= max_chunks:
                        return
                chunk = VideoChunk(index, [], [])
            target = index * chunk_seconds + len(chunk.images) * (
                chunk_seconds / frames_per_chunk
            )
            if len(chunk.images) < frames_per_chunk and timestamp + 1e-6 >= target:
                picture = frame.to_image().convert("RGB")
                picture.thumbnail((max_side, max_side))
                chunk.timestamps.append(timestamp)
                chunk.images.append(picture)
        if chunk is None:
            raise ValueError(f"no decodable video frames: {path}")
        yield chunk
