import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ttt_frame import codex_teacher
from ttt_frame.teacher_bridge import ANALYSIS_SCHEMA, PACKET_SCHEMA


def _digest_without(value, field):
    body = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _packet(tmp_path: Path):
    packet = tmp_path / "packet"
    frame = packet / "frames" / "chunk-000000" / "frame-0000.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"verified image bytes")
    frame_ref = frame.relative_to(packet).as_posix()
    manifest = {
        "schema": PACKET_SCHEMA,
        "source_name": "scene.mp4",
        "source_sha256": "source-sha256",
        "sampling": {
            "chunk_seconds": 1.0,
            "frames_per_chunk": 1,
            "max_side": 32,
            "max_chunks": 1,
        },
        "chunks": [
            {
                "chunk_index": 0,
                "start_sec": 0.25,
                "end_sec": 0.25,
                "frames": [
                    {
                        "file": frame_ref,
                        "timestamp_sec": 0.25,
                        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                        "width": 32,
                        "height": 24,
                    }
                ],
            }
        ],
    }
    manifest["manifest_sha256"] = _digest_without(manifest, "manifest_sha256")
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return packet, manifest, frame


def _analysis(manifest):
    frame_ref = manifest["chunks"][0]["frames"][0]["file"]
    return {
        "schema": ANALYSIS_SCHEMA,
        "packet_manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": manifest["source_sha256"],
        "segments": [
            {
                "chunk_index": 0,
                "start_sec": 0.25,
                "end_sec": 0.25,
                "frame_refs": [frame_ref],
                "summary": "A cup is visible.",
                "observations": ["At 0.25 seconds, a cup is visible."],
                "entities": [
                    {
                        "id": "cup-1",
                        "type": "object",
                        "name": "cup",
                        "attributes": ["white"],
                    }
                ],
                "events": [],
                "relations": [],
                "uncertainty": [],
                "qa": [{"question": "What is visible?", "answer": "A cup."}],
            }
        ],
        "global_summary": "A cup remains visible.",
    }


def _assert_closed_objects(schema):
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for child in schema["properties"].values():
            _assert_closed_objects(child)
    elif schema.get("type") == "array":
        _assert_closed_objects(schema["items"])


def test_analysis_schema_is_closed_and_bound_to_packet():
    manifest = {"manifest_sha256": "packet-hash", "source_sha256": "source-hash"}

    schema = codex_teacher.analysis_schema(manifest)

    _assert_closed_objects(schema)
    properties = schema["properties"]
    assert properties["schema"]["enum"] == [ANALYSIS_SCHEMA]
    assert properties["packet_manifest_sha256"]["enum"] == ["packet-hash"]
    assert properties["source_sha256"]["enum"] == ["source-hash"]
    segment = properties["segments"]["items"]
    assert "qa" in segment["required"]
    assert set(segment["properties"]["qa"]["items"]["required"]) == {
        "question",
        "answer",
    }


def test_analyze_packet_rejects_missing_chatgpt_login(tmp_path, monkeypatch):
    packet, _, _ = _packet(tmp_path)
    destination = tmp_path / "analysis"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Not logged in")

    monkeypatch.setattr(codex_teacher.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Codex ChatGPT login required"):
        codex_teacher.analyze_packet(packet, destination)

    assert [call[0] for call in calls] == [["codex", "login", "status"]]
    assert not destination.exists()


def test_analyze_packet_mocked_codex_success_writes_auditable_result(tmp_path, monkeypatch):
    packet, manifest, frame = _packet(tmp_path)
    destination = tmp_path / "analysis"
    expected_analysis = _analysis(manifest)
    calls = []
    exec_context = {}

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="Logged in using ChatGPT\n", stderr=""
            )
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 9.9.9\n", stderr="")
        assert command[1] == "exec"
        raw_path = Path(command[command.index("--output-last-message") + 1])
        raw_path.write_text(json.dumps(expected_analysis), encoding="utf-8")
        kwargs["stdout"].write('{"type":"mock.completed"}\n')
        kwargs["stdout"].flush()
        exec_context.update(
            command=command,
            prompt=kwargs["input"],
            cwd=Path(kwargs["cwd"]),
            cwd_existed=Path(kwargs["cwd"]).is_dir(),
        )
        return subprocess.CompletedProcess(command, 0)

    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(codex_teacher.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_teacher.time, "perf_counter", lambda: next(ticks))

    report = codex_teacher.analyze_packet(
        packet,
        destination,
        model="gpt-test",
        effort="high",
        timeout=45,
    )

    assert len(calls) == 3
    command = exec_context["command"]
    assert command[0:2] == ["codex", "exec"]
    assert command[-1] == "-"
    assert command[command.index("--model") + 1] == "gpt-test"
    assert command[command.index("--image") + 1] == str(frame.resolve())
    assert "--ignore-user-config" in command
    assert "shell_tool" in command and "multi_agent" in command and "plugins" in command
    assert exec_context["cwd_existed"]
    assert exec_context["cwd"].name.startswith("ttt-codex-teacher-")
    assert exec_context["prompt"] == (destination / "prompt.txt").read_text(encoding="utf-8")

    assert json.loads((destination / "analysis.json").read_text(encoding="utf-8")) == expected_analysis
    assert json.loads((destination / "raw.json").read_text(encoding="utf-8")) == expected_analysis
    assert (destination / "events.jsonl").read_text(encoding="utf-8") == (
        '{"type":"mock.completed"}\n'
    )
    assert (destination / "stderr.log").read_text(encoding="utf-8") == ""
    assert report["backend"] == "codex_cli"
    assert report["auth"] == "ChatGPT"
    assert report["cli_version"] == "codex-cli 9.9.9"
    assert report["requested_model"] == "gpt-test"
    assert report["reasoning_effort"] == "high"
    assert report["images"] == 1
    assert report["elapsed_seconds"] == 2.5
    assert json.loads((destination / "run.json").read_text(encoding="utf-8")) == report


def test_openai_compatible_teacher_normalizes_origin_and_records_usage(tmp_path, monkeypatch):
    from ttt_frame import api_teacher

    packet, manifest, _ = _packet(tmp_path)
    expected = _analysis(manifest)
    calls = {}

    class Completions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                    content=json.dumps(expected)))],
                usage=types.SimpleNamespace(prompt_tokens=111, completion_tokens=23,
                                            total_tokens=134),
            )

    class Client:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    monkeypatch.setenv("TTT_TEACHER_API_KEY", "secret-do-not-log")
    destination = tmp_path / "api"
    report = api_teacher.analyze_packet(packet, destination, model="vision-test",
                                        base_url="https://starwithcoding.com")
    assert calls["client"] == {"api_key": "secret-do-not-log",
                                "base_url": "https://starwithcoding.com/v1", "timeout": 180.0}
    assert calls["model"] == "vision-test"
    assert calls["response_format"]["type"] == "json_schema"
    assert report["usage"] == {"prompt_tokens": 111, "completion_tokens": 23,
                                "total_tokens": 134}
    assert "secret-do-not-log" not in (destination / "run.json").read_text()
    assert json.loads((destination / "analysis.json").read_text()) == expected
