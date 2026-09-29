# Codex-derived MCQ sanity evaluation

`data/q9_codex_mcq.jsonl` contains 19 short questions made from the saved
Codex scene analyses for the `supermemory-9` kitchen recording. The memory
environment itself is the complete 18-clip recording (`0--1080` seconds). The
Codex analyses are sampled-frame observations, so each question keeps its
source clip and local/global evidence span and avoids uncertain claims such as
whether an unseen action was completed.

The answer letters are deliberately balanced: A=5, B=5, C=5, D=4. A model
that always emits the most frequent letter therefore scores 5/19 (26.3%). The
gold and evidence fields stay in the evaluator; only the question, options,
and the single-letter instruction are sent to a model.

## Main three-method run

The three methods requested for the first comparison were the existing local
teacher LoRA, Codex teacher LoRA, and dense 4-second/8-frame Spatial-TTT:

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_codex_mcq.py \
  --items-path data/q9_codex_mcq.jsonl \
  --parallel --devices cuda:3,cuda:4,cuda:5 \
  --max-new-tokens 16 \
  --output runs/q9_codex_mcq_models.json
```

The run produced the following exact-MCQ scores:

| method | overall | location | relation | placement | order | long-range |
|---|---:|---:|---:|---:|---:|---:|
| LoRA + local teacher | 11/19 (57.9%) | 1/4 | 3/5 | 2/2 | 3/4 | 2/4 |
| LoRA + Codex teacher | 10/19 (52.6%) | 0/4 | 2/5 | 2/2 | 3/4 | 3/4 |
| Spatial-TTT, 4s/8f | 14/19 (73.7%) | 3/4 | 5/5 | 2/2 | 2/4 | 2/4 |

The raw answers and parsed letters are stored in the local output JSON under
`runs/` (that directory is intentionally git-ignored).

## Spatial paired controls

For the same questions, adding the older Spatial memory and the official
checkpoint with the fast-memory read disabled gives:

If “three models” is intended to mean the three Spatial conditions themselves,
the equivalent batch command is:

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_spatial_items.py \
  --items-path data/q9_codex_mcq.jsonl \
  --parallel --devices cuda:3,cuda:4,cuda:5 \
  --max-new-tokens 16 \
  --output runs/q9_codex_mcq_spatial_three_conditions.json
```

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_codex_mcq.py \
  --items-path data/q9_codex_mcq.jsonl \
  --include-sparse-spatial --include-spatial-control \
  --parallel --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
  --max-new-tokens 16 \
  --output runs/q9_codex_mcq_all_conditions.json
```

| method | overall | location | relation | placement | order | long-range |
|---|---:|---:|---:|---:|---:|---:|
| Spatial-TTT, 4s/8f | 14/19 (73.7%) | 3/4 | 5/5 | 2/2 | 2/4 | 2/4 |
| Spatial-TTT, 60s/16f | 12/19 (63.2%) | 3/4 | 4/5 | 2/2 | 2/4 | 1/4 |
| Spatial checkpoint, `--without-memory` | 12/19 (63.2%) | 2/4 | 5/5 | 2/2 | 2/4 | 1/4 |

Relative to the paired no-memory control, dense Spatial memory corrected the
refrigerator-to-onion question, the late island/stovetop sequence, and the
shared-cutting-board relation, but changed one previously correct sequence
answer to an error. It did not fix the blue-plate, dark-bowl, or several
long-range location questions. This is evidence that the fast weights alter
the readout; it is not evidence that every retrieved fact is accurate.

The four `long_range_*` rows are the most relevant to the 1080-second memory
claim. Their scores were local LoRA 2/4, Codex LoRA 3/4, dense Spatial 2/4,
sparse Spatial 1/4, and the no-memory control 1/4. The local and dense scores
on the full set are therefore driven partly by short-range layout questions;
they should not be described as long-term recall accuracy.

## Interpretation limits

These questions were authored from Codex analyses, including the same Codex
teacher family used to create the Codex LoRA memory. They are therefore a
targeted sanity check rather than an independent benchmark, and the 19 items
are too small for a capability claim. Several static kitchen-layout questions
can be answered from the pretrained model's visual or language prior. A stronger
follow-up should hold out question facts, permute options for the same fact,
and add questions authored from an independent annotation or a different
video.
