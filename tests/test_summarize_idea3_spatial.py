"""Archive only frozen, checksummed Idea 03 reports without reading benchmark answers."""
import pytest

from scripts import evaluate_idea3_spatial as evaluator
from scripts import summarize_idea3_spatial as runner
from test_evaluate_idea3_spatial import make_inputs


@pytest.fixture
def scored_run(tmp_path):
    scores, split, root, output, baseline = make_inputs(tmp_path)
    evaluator.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    evaluator.score(scores, split, root, output, baseline, inference_protocol=scores.with_suffix(".protocol.json"))
    (root / "answers.jsonl").unlink()
    return output, tmp_path / "archive", scores


def test_archive_preserves_complete_scores_predictions_and_integrity(scored_run):
    run, archive, scores = scored_run
    manifest = runner.archive(run, archive)
    assert manifest["cases"] == 5
    assert manifest["answer_files_read_by_archiver"] is False
    assert (archive / "scores.jsonl").read_bytes() == scores.read_bytes()
    for name, metadata in manifest["files"].items():
        assert (archive / name).stat().st_size == metadata["bytes"]
        assert runner.sha256_file(archive / name) == metadata["sha256"]
    assert set(manifest["files"]) == {p.name for p in archive.iterdir()} - {"manifest.json"}
    assert "silver labels" in (archive / "results.md").read_text()
    assert "correct_answer" not in (archive / "per_case.jsonl").read_text()


@pytest.mark.parametrize("name", ["summary.json", "per_case.jsonl", "per_case.csv"])
def test_archive_rejects_corrupt_reports(scored_run, name):
    run, archive, _ = scored_run
    with (run / "report" / name).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        runner.archive(run, archive)
    assert not archive.exists()


def test_archive_requires_original_score_checksum(scored_run):
    run, archive, scores = scored_run
    with scores.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="scores changed"):
        runner.archive(run, archive)


def test_archive_cannot_overwrite_prior_results(scored_run):
    run, archive, _ = scored_run
    runner.archive(run, archive)
    with pytest.raises(FileExistsError):
        runner.archive(run, archive)


def test_archive_binds_selection_sidecar(scored_run):
    run, archive, _ = scored_run
    selection = runner.read_json(run / "validation_selection.json")
    selection["search"]["beta"] = 999
    runner.write_json(run / "validation_selection.json", selection)
    with pytest.raises(ValueError, match="selection sidecar"):
        runner.archive(run, archive)


def test_archive_remains_portable_without_current_inference_sources(scored_run, monkeypatch):
    run, archive, _ = scored_run
    # The archiver checks saved protocol bytes, not the machine's current implementation.
    monkeypatch.setattr(evaluator, "validate_inference_protocol", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("must not require live inference code")))
    runner.archive(run, archive)
    assert (archive / "inference_protocol.json").exists()
    assert (archive / "split_manifest.json").exists()


def test_archive_output_cannot_be_inside_source_benchmark(scored_run):
    run, _, _ = scored_run
    protocol = runner.read_json(run / "protocol.json")["protocol"]
    output = __import__("pathlib").Path(protocol["benchmark_root"]) / "archive"
    with pytest.raises(ValueError, match="outside"):
        runner.archive(run, output)
