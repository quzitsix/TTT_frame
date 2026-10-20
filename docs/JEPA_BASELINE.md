# V-JEPA 2 baseline for experiment C

> Spatial-TTT、V-JEPA、StreamTTT、显式/递归记忆和其他路线的统一比较，以及新的
> 非 counting 实验 C 设计见
> [`VIDEO_MEMORY_ARCHITECTURE_COMPARISON.md`](VIDEO_MEMORY_ARCHITECTURE_COMPARISON.md)。

## Current status

The formal result now uses `experiment_c_relocation_v2`. Every episode and condition contains exactly 16 observed cards, and `future_event` is stored only as a target. It is never rendered, encoded, or used as an input feature.

The comparison uses Meta's official frozen [V-JEPA 2](https://github.com/facebookresearch/vjepa2) ViT-L encoder. V-JEPA 2 has no language decoder or persistent test-time state, so it cannot directly answer either the open or multiple-choice C questions. The implemented baseline is explicitly:

```text
16 chronological observed cards
        │  letterbox to 256 × 256, ImageNet normalization
        ▼
official frozen V-JEPA 2 ViT-L/16 encoder
        │  8 tubelet states × 1024 dimensions
        ▼
spatial mean per tubelet, concatenate time → 8192-D feature ──┐
                                                               ├─→ group-excluded
canonical observed event list → SHA-256 content-group ID ──────┘   kernel-ridge readout
```

For each test episode, the readout excludes every episode whose complete observed event list is identical to that test episode. This matters because the 12-episode fixture contains exact repeated streams: an ordinary leave-one-episode-out split could put the same observation sequence in both train and test. The corrected split has 10 content groups for `relocated`, 10 for `same_scene_control`, and 9 for `filler_only_control`; all 12 episodes are still evaluated. This is a supervised representation probe. It is neither TTT nor evidence that V-JEPA 2 itself supplies long-term memory.

The same content-group split and readout procedure is applied to a `16 frames × 16 × 16 RGB` pixel feature. This is an important control because the PNG cards contain literal text, repeated layouts, and deterministic generator structure.

## Official checkpoint provenance

The server already contained the official source and checkpoint, so no model download was needed:

- Source: `/data/leiyj23/ICRA2026/assets/vjepa2-main-full`
- Checkpoint: `/data/leiyj23/ICRA2026/models/vjepa2/vitl.pt`
- Architecture: V-JEPA 2 ViT-L/16, FPC64-256
- Encoder parameters: 303,885,312
- Checkpoint size: 5,127,726,842 bytes
- Local SHA-256: `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`

The checkpoint was verified against Meta's [official `vitl.pt` asset](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt). The official object and local file both have multipart ETag `71cf6756d55670a39bb369fca4716a54-612`; the local value was recomputed using the object's 8 MiB multipart layout. This verifies the complete binary rather than only its filename or size.

The `meowbench` environment has `timm 1.0.30` and `einops 0.8.2`. They are also listed in the `jepa` optional dependency in `pyproject.toml`. The runner imports the external official source read-only.

## Formal C-PNG protocol (fixture schema v2)

- Data: 12 episodes and five query types, evaluated separately under `relocated`, `same_scene_control`, and `filler_only_control`.
- Observation budget: all 16 observed cards in every episode and condition.
- Future boundary: `future_event_observed=false` in every result. The `next_action` target is never an input card.
- Controls: `condition_stream` creates matched 16-card condition streams. Relocated cards are read only from manifest `frame_paths`; no filesystem glob is used. The two counterfactual controls are rendered in memory from their transformed event streams.
- Preprocessing: V-JEPA uses aspect-preserving letterbox to `256 × 256` followed by ImageNet normalization. The pixel control starts from the same letterboxed images, resizes them to `16 × 16`, and scales RGB by `1/255` without ImageNet normalization.
- Readout: centered kernel ridge with `lambda=0.1`, fitted separately for location, mover, owner, state, and next action.
- Validation: hash each canonical observed event list, then leave out the entire matching content group. Candidate labels and the fitted readout come only from episodes in other groups. This prevents an exact duplicate observation stream from crossing the train/test boundary.

In each of relocated and same-scene, six held-out items have a correct label that is absent from their group-excluded training fold. The report therefore includes both all-item accuracy and accuracy on folds whose answer was seen in training.

## Formal C-PNG results

| condition | arm | exact | accuracy | seen-label exact | seen-label accuracy | prior expected |
|---|---|---:|---:|---:|---:|---:|
| relocated | V-JEPA 2 + readout | 38/60 | 0.6333 | 38/54 | 0.7037 | 0.2240 |
| relocated | pixels + same readout | 40/60 | 0.6667 | 40/54 | 0.7407 | 0.2240 |
| same scene | V-JEPA 2 + readout | 35/60 | 0.5833 | 35/54 | 0.6481 | 0.2240 |
| same scene | pixels + same readout | 40/60 | 0.6667 | 40/54 | 0.7407 | 0.2240 |
| filler only | V-JEPA 2 + readout | 53/60 | 0.8833 | 53/60 | 0.8833 | 0.6946 |
| filler only | pixels + same readout | 52/60 | 0.8667 | 52/60 | 0.8667 | 0.6946 |

Across all three conditions, V-JEPA 2 scores 126/180 = **0.7000**, while the matched pixel readout scores 132/180 = **0.7333**. Pixels are higher for relocated and same-scene; V-JEPA 2 is one answer higher for filler-only. Neither frozen feature arm contains a persistent online state.

| condition | arm | location | mover | owner | state | next action |
|---|---|---:|---:|---:|---:|---:|
| relocated | V-JEPA 2 | 1.0000 | 0.2500 | 0.5000 | 1.0000 | 0.4167 |
| relocated | pixels | 1.0000 | 0.3333 | 0.5000 | 1.0000 | 0.5000 |
| same scene | V-JEPA 2 | 1.0000 | 0.1667 | 0.3333 | 1.0000 | 0.4167 |
| same scene | pixels | 1.0000 | 0.3333 | 0.5000 | 1.0000 | 0.5000 |
| filler only | V-JEPA 2 | 1.0000 | 1.0000 | 0.4167 | 1.0000 | 1.0000 |
| filler only | pixels | 1.0000 | 1.0000 | 0.3333 | 1.0000 | 1.0000 |

The relocated-minus-same-scene difference is `+0.0500` for V-JEPA 2 and `0.0000` for pixels. For V-JEPA 2 this is three answers out of 60, split between mover and owner. With only 12 episodes and deterministic generator cycles, this is not evidence that relocation helps the representation.

The high filler-only score is also not a memory result. Mover, state, and next action are constant `unknown/intact/unknown` controls, which makes the deterministic majority predictor score 0.6000 and raises the group-aware sampled-label-prior expectation to 0.6946. The variable owner field is only 0.4167 for V-JEPA 2 and 0.3333 for pixels.

The bf16 encoder allocates 608,819,200 bytes and peaks at 675,010,560 bytes for the 16-frame run. The current formal artifacts record 3.57–4.83 seconds to encode all 12 clips after model construction/checkpoint loading on an RTX 4090 D.

## Interpretation

The deployment is valid: the runner loads a byte-verified official target encoder, extracts nonconstant features, excludes the held-out episode and every exact-content duplicate from the readout fit, covers all three matched conditions, and records fold-level outputs.

The corrected formal C-PNG result shows no aggregate V-JEPA 2 advantage over the pixel control. Exact-stream grouping removes the direct duplicate leak, but perfect location and state recovery remains a fixture shortcut: `moved_object`, final location, and final state follow the same three-episode cycle. The literal text, repeated card layout, and deterministic generator therefore still expose labels without requiring persistent memory. The next-action target is properly withheld, but it is deterministically related to visible owner and final-location variables; a supervised readout can exploit that relation without learning a general future world model.

Two condition artifacts further limit interpretation. Filler cards visibly contain the internal strings `control distractor`, `actor: control`, and `control_records: 4`. The relocated/same-scene transform also changes `living room shelf` to `shelf` and `side table` to `table`, so that contrast changes wording as well as the scene/background cue. These artifacts must be removed in fixture schema v3 or C-temporal.

This result is therefore useful as a deployment and representation sanity check. It cannot rank V-JEPA 2 against Spatial-TTT as a long-term memory system because the frozen JEPA arm contains no persistent online state, and because low-resolution pixels already expose enough template structure to do better.

A fair next JEPA comparison should use rendered or real household video, split by scene/house/object identity, remove deterministic answer cycles, and train a fixed readout on a larger training set. Two distinct arms should then be kept:

1. `frozen V-JEPA 2 + fixed readout` to measure predictive representation quality;
2. `V-JEPA 2 + recurrent or fast-weight state` to measure persistent online memory under the same observation and query budget as Spatial-TTT.

V-JEPA 2-AC is appropriate only when candidate actions or control inputs are available. Its action-conditioned predictor addresses a different problem from passive first-person memory.

## Legacy diagnostics

The earlier fixture schema v2 table used ordinary leave-one-episode-out validation. It scored 142/180 = 0.7889 for V-JEPA 2 and 148/180 = 0.8222 for pixels. That split excluded only the held-out episode ID even when another episode had the exact same observed event stream, so identical observations could appear in training. Those values are now leakage diagnostics and must not be reported as held-out representation performance. The three `experiment_c_jepa_v2_*` artifact paths in the reproduction section now contain the corrected content-group results; the old LOEO values are retained here only to explain the change.

The legacy fixture schema v1 produced 0.8000 for both V-JEPA 2 and pixels when 16 frames were sampled from its 17-card stream, which included the explicit forecast target. An earlier `pre_forecast` run gave 0.8000 for V-JEPA 2 and 0.7833 for pixels after removing that target card. Those files remain at:

- `runs/experiment_c_jepa_12ep_16f.json`
- `runs/experiment_c_jepa_pre_forecast_12ep_16f.json`

They are legacy plumbing results and must not be mixed with the formal content-group C-PNG table for fixture schema v2. The current runner always filters a legacy `forecast` event before encoding.

## Reproduction

Install the project with the JEPA dependency group if needed:

```bash
conda run --no-capture-output -n meowbench \
  python -m pip install -e '.[jepa]'
```

Run the formal relocated condition on physical GPU 7:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n meowbench \
  python scripts/evaluate_experiment_c_jepa.py \
  --data data/experiment_c/manifest.json \
  --condition relocated \
  --clip-scope pre_forecast \
  --episodes 12 \
  --clip-frames 16 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output runs/experiment_c_jepa_v2_relocated_12ep_16f.json \
  --feature-cache runs/experiment_c_jepa_v2_relocated_12ep_16f_features.pt
```

Run the matched same-scene condition by changing three arguments:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n meowbench \
  python scripts/evaluate_experiment_c_jepa.py \
  --data data/experiment_c/manifest.json \
  --condition same_scene_control \
  --clip-scope pre_forecast \
  --episodes 12 \
  --clip-frames 16 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output runs/experiment_c_jepa_v2_same_scene_12ep_16f.json \
  --feature-cache runs/experiment_c_jepa_v2_same_scene_12ep_16f_features.pt
```

Run the filler-only condition:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n meowbench \
  python scripts/evaluate_experiment_c_jepa.py \
  --data data/experiment_c/manifest.json \
  --condition filler_only_control \
  --clip-scope pre_forecast \
  --episodes 12 \
  --clip-frames 16 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output runs/experiment_c_jepa_v2_filler_only_12ep_16f.json \
  --feature-cache runs/experiment_c_jepa_v2_filler_only_12ep_16f_features.pt
```

Content grouping is automatic; no additional split flag is required. Re-running an unchanged command reuses its encoder feature cache and recomputes the group-excluded readout. The cache fingerprint covers the manifest, relocated input PNG bytes, checkpoint bytes, the official Python source tree, renderer font, runner, and preprocessing configuration. Pass `--refresh-cache` after an untracked environment change. `--pixel-only` is an independent CPU-only pipeline check and does not require the external JEPA source or checkpoint. In each output, verify `protocol.readout` is `leave-one-content-group-out centered kernel ridge` and inspect `protocol.content_groups` (10, 10, and 9 for relocated, same-scene, and filler-only).

The tracked code and document are `scripts/evaluate_experiment_c_jepa.py` and this file. JSON outputs and feature caches under `runs/` are local artifacts ignored by Git.
