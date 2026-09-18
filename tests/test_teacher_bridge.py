import json
from pathlib import Path

import pytest
from PIL import Image

from ttt_frame.teacher_bridge import (
    ANALYSIS_SCHEMA,
    PACKET_SCHEMA,
    canonicalize_analysis,
    prepare_teacher_packet,
    validate_teacher_analysis,
)


def _video(path: Path):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width, stream.height, stream.pix_fmt = 24, 16, "yuv420p"
        for i in range(6):
            frame = av.VideoFrame.from_image(Image.new("RGB", (24, 16), (i * 20, 20, 40)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_prepare_packet_and_validate_analysis(tmp_path):
    source = tmp_path / "scene.mp4"
    _video(source)
    manifest = prepare_teacher_packet(source, tmp_path / "packet", chunk_seconds=1,
                                      frames_per_chunk=2, max_side=32, max_chunks=1)
    assert manifest["schema"] == PACKET_SCHEMA
    assert (tmp_path / "packet" / "teacher_prompt.md").is_file()
    chunk = manifest["chunks"][0]
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "packet_manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": manifest["source_sha256"],
        "segments": [{
            "chunk_index": chunk["chunk_index"], "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"], "frame_refs": [chunk["frames"][0]["file"]],
            "summary": "A visible room.", "observations": ["A colored object is visible."],
            "entities": [{"id": "object-1", "type": "object", "name": "object", "attributes": []}],
            "events": [], "relations": [], "uncertainty": [],
        }],
        "global_summary": "A room is visible.",
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    validated = validate_teacher_analysis(analysis_path, manifest)
    text = canonicalize_analysis(validated)
    assert "Observation: A colored object" in text


def test_analysis_rejects_foreign_packet_and_future_event(tmp_path):
    source = tmp_path / "scene.mp4"
    _video(source)
    manifest = prepare_teacher_packet(source, tmp_path / "packet", chunk_seconds=1,
                                      frames_per_chunk=2, max_side=32)
    chunk = manifest["chunks"][0]
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "packet_manifest_sha256": "wrong", "source_sha256": manifest["source_sha256"],
        "segments": [{"chunk_index": chunk["chunk_index"], "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"], "frame_refs": [chunk["frames"][0]["file"]],
            "summary": "Room", "observations": [], "entities": [],
            "events": [], "relations": [], "uncertainty": []}],
        "global_summary": "Room",
    }
    p = tmp_path / "analysis.json"
    p.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        validate_teacher_analysis(p, manifest)
