# Frozen bench_v0 MCQ baseline artifacts

This directory preserves the baseline for 200 six-choice forecasting questions.
See [the experiment report](../../docs/BENCH_V0_MCQ_BASELINES.md) for architecture
definitions, limitations, results and commands.

- `protocol.json`: frozen input/model/source hashes and execution settings.
- `research_day_split.json`: fixed unofficial 111/32/57 day-based split.
- `aggregate_metrics.json`: machine-readable aggregate and paired scores, with
  per-case answer keys omitted.
- `*_predictions.jsonl`: nine standard ID/prediction exports, 200 rows each.
- `prediction_diagnostics.json`: past-frame identity and fast/slow/base logit
  checks; no benchmark answers are used by this diagnostic.
- `vision_patch_embed_validation.json` and `smoke_execution_comparison.json`:
  execution-path validation and its numerical limitations.
- `execution_environment.json`: package versions, CUDA build and GPU identity.
- `memory_state_audit.json`: actual fast-weight changes and exact reset check
  from replaying the first case, without answering or loading answer keys.
- `artifact_manifest.json`: checksums of all other files in this archive.

Original per-case logits, timing, sampling and write logs are retained locally
under `runs/forecast_bench_v0/mcq_baseline_v1/`. Videos, questions, answer keys,
checkpoints and generated frame caches are not copied into this directory.

The base Git revision inside the protocol predates the implementation commit;
the protocol's per-file SHA-256 hashes identify the actual inference source.
Keep the protocol fingerprint with any copied predictions. Later model training
must use the same fixed test IDs for comparisons with the 57-case test scores.
