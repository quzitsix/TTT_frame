# Experiment C: controlled relocation and rearrangement

> This is a synthetic mechanism probe. It is not a result on real household video.

The fixture writes object–location, object–owner, mover, state and next-action events. `relocated` and `same_scene_control` contain the same number of writes; the query cue has a different scene component only in the former. `filler_only_control` removes the move to measure ordinary retention. `explicit_retrieval` is a fixed-size table control, and `oracle` is a ceiling.

- Episodes: **12**; trials: **3**
- Semantic code dimension: **64**; scene drift strength: **0.35**
- Rows: **2160**; no external video or model download was used

## Accuracy by arm and condition

| arm | condition | n | coverage | accuracy |
|---|---|---:|---:|---:|
| explicit_retrieval | filler_only_control | 180 | 1.000 | 0.5333 |
| explicit_retrieval | relocated | 180 | 1.000 | 1.0 |
| explicit_retrieval | same_scene_control | 180 | 1.000 | 1.0 |
| lact | filler_only_control | 180 | 1.000 | 0.2389 |
| lact | relocated | 180 | 1.000 | 0.8833 |
| lact | same_scene_control | 180 | 1.000 | 0.5611 |
| none | filler_only_control | 180 | 0.000 | — |
| none | relocated | 180 | 0.000 | — |
| none | same_scene_control | 180 | 0.000 | — |
| oracle | filler_only_control | 180 | 1.000 | 1.0 |
| oracle | relocated | 180 | 1.000 | 1.0 |
| oracle | same_scene_control | 180 | 1.000 | 1.0 |

## Relocated minus same-scene accuracy

A negative value means the scene change hurt this query type under the chosen cue drift. It is a diagnostic, not a claim about real video.

| arm | query type | delta |
|---|---|---:|
| oracle | location | 0.0 |
| oracle | mover | 0.0 |
| oracle | owner | 0.0 |
| oracle | state | 0.0 |
| oracle | next_action | 0.0 |
| explicit_retrieval | location | 0.0 |
| explicit_retrieval | mover | 0.0 |
| explicit_retrieval | owner | 0.0 |
| explicit_retrieval | state | 0.0 |
| explicit_retrieval | next_action | 0.0 |
| lact | location | 1.0 |
| lact | mover | 0.0 |
| lact | owner | -0.0556 |
| lact | state | 0.0 |
| lact | next_action | 0.6667 |
| none | location | — |
| none | mover | — |
| none | owner | — |
| none | state | — |
| none | next_action | — |

## Interpretation boundary

The codebook removes OCR and visual encoder confounds so this run answers only whether a fixed-size associative state can retain bindings through filler and a scene-key change. A VLM follow-up should render the same event manifest into a video and repeat the conditions with equal frame budgets. `none` has zero coverage by design; it is a blind control, not a failed model run.

## Visual PNG-card follow-up

As a separate sanity check, one episode (five queries) was rendered as PNG event cards and passed through the native Qwen3-VL interfaces with all 17 chronological cards. The cards are text-heavy synthetic stimuli, so this is a visual plumbing check rather than a household-video result.

| arm | queries | mean token F1 | exact | gold substring |
|---|---:|---:|---:|---:|
| ICL visual | 5 | 0.3455 | 1 | 1 |
| Spatial-TTT | 5 | 0.0182 | 0 | 0 |

ICL answered one query exactly, while Spatial-TTT did not. The tiny, text-heavy card sample is not enough to establish a visual-memory advantage; the symbolic C numbers must not be interpreted as VLM accuracy. The next C iteration should replace the cards with photorealistic rendered scenes or real controlled video while keeping the same query manifest.
