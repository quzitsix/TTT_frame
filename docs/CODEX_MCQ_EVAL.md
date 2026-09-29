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

## Qwen3.5 frozen baseline

The server also has `/data/hf_models/Qwen/Qwen3.5-35B-A3B`.  Its 35B-total,
3B-active hybrid MoE architecture is not a drop-in replacement for the
Qwen3-VL-2B model used by the saved memories: the checkpoint has
`model_type=qwen3_5_moe`, while `SpatialQwenMemory` currently requires
`qwen3_vl`.  The existing fast-weight and LoRA checkpoints therefore cannot
be loaded into Qwen3.5.  The model README requires a recent Transformers main
build; the server's separate `qwen35-env` provides that runtime and places the
BF16 weights over four GPUs.

For a capacity/control comparison, Qwen3.5 was served with reasoning disabled
and received the same 19 text-only prompts as the other methods.  No video,
gold answer, Codex evidence, or saved TTT memory was sent to it:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup \
  /data/quzitsix/qwen35-env/bin/transformers serve \
  /data/hf_models/Qwen/Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port 18000 --device auto --dtype bfloat16 \
  --no-continuous-batching --reasoning off --log-level info \
  >/tmp/qwen35-serve.log 2>&1 &

conda run --no-capture-output -n meowbench python scripts/evaluate_mcq_api.py \
  --items-path data/q9_codex_mcq.jsonl \
  --base-url http://127.0.0.1:18000/v1 \
  --model /data/hf_models/Qwen/Qwen3.5-35B-A3B \
  --max-tokens 16 \
  --output runs/q9_codex_mcq_qwen35.json
```

It scored **16/19 (84.2%)**: location 3/4, relation 4/5, placement 2/2,
order 4/4, and long-range 3/4.  The Qwen3-VL-2B no-memory control scored
12/19 (63.2%) on the same prompts.  This four-question improvement shows that
the stronger model has a better text-side prior and answer selection ability,
but it is not a memory result: the model never saw the recording.  A direct
Qwen3.5 Spatial-TTT comparison requires a new adapter for its linear-attention
and full-attention layers, plus a new checkpoint trained for its 2048-wide
hidden states.

As a letter-bias check, I reversed the A/B/C/D positions in every item and
remapped the gold letters while keeping the option text unchanged.  Qwen3.5
 scored 17/19 on that variant, so the high score is not explained by always
 choosing one letter.  The stronger explanation is that many distractors are
 implausible or semantically mismatched: a microwave above a stove, produce
 taken from an open refrigerator, and plates or utensils in a dishwasher rack
 are easy language-level completions.  Choosing the longest option alone also
 gets 10/19 on this set.  The remaining Qwen3.5 errors include the dark-bowl
 location, dishwasher contents, and one cross-clip chronology question; these
 are closer to tests of actual visual or long-term recall.

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
