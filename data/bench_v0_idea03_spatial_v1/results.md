# Idea 03: Spatial-TTT address-search forecasting

Exploratory comparison on the fixed bench_v0 day split; silver labels, overlapping windows,
and only two test recording days. Prediction files were fixed before test answers were scored.

| Method | Validation | Test | Test accuracy | vs full ICL gains / losses |
|---|---:|---:|---:|---:|
| prior | 4/32 | 3/57 | 5.26% | 1 / 19 |
| direct | 8/32 | 9/57 | 15.79% | 5 / 17 |
| direct_matched | 8/32 | 9/57 | 15.79% | 5 / 17 |
| search | 6/32 | 6/57 | 10.53% | 3 / 18 |
| reset | 4/32 | 3/57 | 5.26% | 1 / 19 |
| frozen_init | 4/32 | 3/57 | 5.26% | 1 / 19 |
| wrong_direct | 7/32 | 10/57 | 17.54% | 6 / 17 |
| wrong_direct_matched | 7/32 | 10/57 | 17.54% | 6 / 17 |
| wrong_search | 2/32 | 9/57 | 15.79% | 5 / 17 |
| direct_raw | 3/32 | 10/57 | 17.54% | 5 / 16 |
| search_raw | 2/32 | 10/57 | 17.54% | 6 / 17 |
| wrong_raw | 1/32 | 15/57 | 26.32% | 9 / 15 |
| full_icl | 6/32 | 21/57 | 36.84% | — |

Validation selected direct beta=1.0; search radius=0.5, beta=1.0.

`direct` chooses its own beta on validation. `direct_matched` uses the search beta,
so its paired comparison with `search` changes only the read procedure. Reset and
frozen initialization share the same native candidate-likelihood prior; reset association
deltas are exactly zero. Wrong-video controls use the corresponding selected coefficients.

`*_raw` are memory-only diagnostics at the radius selected for the mixed search score;
they are not independently tuned competitors. All score normalization is within question
and permutation invariant. Beta=0 permits validation to reject the memory contribution.

Both full ICL and Spatial use 30 historical plus 60 recent
frames at max side 224, spanning 1800+30 seconds.
Model checkpoint, readout, and query compute differ.
This does not establish an architecture-level win. The official Spatial checkpoint is
used through this repository's parameter-only path; this is not the complete official
hybrid inference workflow or the proposal's outer-trained semantic classifier.
The native text-V candidate target is an untrained semantic surrogate. A last-observed
visual key versus a future-question query is not a same-fact recall comparison; the visual
probe cannot establish that the desired future information was written but unreadable.

Per-case predictions/correctness: `per_case.jsonl` and `per_case.csv`. Raw candidate
regularized association objectives and prior scores: `scores.jsonl`. Pairwise gain/loss IDs and day counts:
`summary.json`. Validation grid and frozen evaluation protocol: `protocol.json`.
Inference model/source identities: `inference_protocol.json`; fixed days: `split_manifest.json`.

No test labels were used for query search or configuration selection. The archiver does
not access answer files. Actual ingestion/query runtime and source/model identities,
when emitted by the scorer, remain in `scores.jsonl`; fixed budget is in `protocol.json`.
