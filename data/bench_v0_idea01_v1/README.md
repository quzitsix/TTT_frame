# Idea 01 forecasting MCQ artifacts

Experiment report: [Idea 01 results and commands](../../docs/IDEA01_BENCH_V0_MCQ_RESULTS.md).

This is a supervised minimal delayed-residual memory prototype using complete
1830-second past prefixes at 2 FPS. The independent training/validation/test
case counts are 111/32/57, grouped by recording day across all participants.
The primary result is the 57-case test score. The 200-case aggregate includes
training and validation examples and is not a held-out benchmark score.

- `aggregate_metrics.json`: split scores, memory-control pairs and historical
  VLM comparisons, with per-case correct answers omitted.
- `*_predictions.jsonl`: eight standard ID/prediction exports, 200 rows each.
- `readout_protocol.json`, `validation_selection.json`: shared-head training,
  fixed candidate regularization grid and validation selection.
- `forecast_protocol.json`: train-only recent predictor and causal online
  memory configuration, plus input/cache/model/source hashes.
- `feature_protocol.json`, `text_protocol.json`: frozen encoder and execution identities.
- `research_day_split.json`: the same day split used for the earlier baselines.
- `execution_diagnostics.json`: measured runtime, parameter changes and latent
  forecast differences; no answer key is used to calculate these diagnostics.
- `artifact_manifest.json`: checksums of the remaining archive files.

The original shared readout, recent predictor, per-case forecast tensors,
candidate embeddings and dense visual caches remain local under
`runs/forecast_bench_v0/idea01_mcq_v1/`. Videos, candidate texts, captions,
answer keys and checkpoints are not copied into this archive.

All seven visual-memory methods share one fitted candidate-text readout.
`text_prior` uses only the train-fitted mean correct-candidate embedding.
`nearest_bank` is explicit visual-feature retrieval, not VLM ICL. Historical
Qwen/Spatial results use 90 frames and no benchmark-trained semantic head;
the Idea 01 prototype uses 3660 frames and supervised training. Cross-family
accuracy comparisons therefore do not establish an equal-budget winner.
