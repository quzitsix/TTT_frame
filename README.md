# TTT_frame

A test-time-training (TTT) memory project, and experiments on **what it can
actually retain**. Two separate baselines now live here: the original LaCT
embedding memory and a generative video-to-LoRA parameter-memory interface.

## VideoQA parameter-memory interface (2026-09-11)

**Real video input is supported; useful long-term recall has not yet been
established.** This new baseline uses test-time LoRA self-distillation, not the
LaCT update and not a reproduction of Spatial-TTT. The original CLIP/LaCT
experiment below is unchanged.

The [real-video pilot report](docs/REPORT_20260911_REAL_VIDEO.md) records the
server evaluation: both LoRA update budgets scored 3/5 on a tiny SuperMemory
subset versus 1/5 blind, while EPIC free-form readout still repeated or invented
content. Those five overlapping-context questions do not establish reliable
long-term recall or relocation robustness.

```
chronological video sessions
  -> bounded RGB chunks (sequential PyAV decoding)
  -> frozen VLM observations + temporary self-generated QA
  -> assistant-token cross-entropy updates to language q_proj/v_proj LoRA
  -> discard frames, observations, training examples, gradients and optimizer
  -> question (+ options, if supplied) -> adapted VLM -> generated answer
```

Only LoRA parameters carry episodic content across the ingestion/query boundary.
The teacher always disables the adapter; it never receives benchmark questions,
options or gold labels. Video observations and QA are temporary training data,
not a query-time retrieval store. This is a **visual-to-text bottleneck**: facts
omitted or hallucinated by the teacher cannot be fixed by parameter storage.
There is no face identification, cross-camera person tracking, or ownership
inference module. Seeing someone hold an object is not proof they own it.

### Install in the existing server conda environment

```bash
conda activate meowbench
cd ~/TTT_frame
python -m pip install -e ".[video]"
```

No separate conda environment is required. The `video` extra adds PEFT, PyAV,
Accelerate and Safetensors while preserving `transformers>=4.57,<5`; the basic
`ttt-frame` dependency remains just PyTorch. Locally tested with torch
2.11.0+cu126, transformers 4.57.6, PEFT 0.14.0, Accelerate 1.13.0 and PyAV 17.0.0.
These are observations, not a requirement to replace the server's working torch.
Use one GPU with bf16 or float32; automatic CPU/disk offload, quantized training,
DDP and custom `trust_remote_code` models are not implemented in this baseline.

### Input videos, then answer from saved parameters

Replace the video paths with chronological recordings of **one environment**.
Use the same base model weights/revision for both commands.

```bash
MODEL=/data/quzitsix/models/Qwen3-VL-2B-Instruct
python -m ttt_frame.videoqa ingest \
  --model-path "$MODEL" --local-files-only \
  --video /path/to/day1.mp4 /path/to/day2.mp4 \
  --chunk-seconds 30 --frames-per-chunk 4 \
  --steps-per-chunk 12 --teacher-max-new-tokens 512 \
  --save memories/home1

# Separate process: needs only base weights + memory, not the videos.
python -m ttt_frame.videoqa ask \
  --memory memories/home1 --model-path "$MODEL" --local-files-only \
  --question "Who last handled the red mug, and where was it placed?"

# Causal read control: same question and base, with LoRA disabled.
python -m ttt_frame.videoqa ask \
  --memory memories/home1 --model-path "$MODEL" --local-files-only \
  --without-memory --question "Who last handled the red mug, and where was it placed?"
```

The save directory must be new. It contains only `adapter.safetensors` and
`memory.json` (configuration + numeric counters); it contains no video paths,
captions, frame embeddings, training samples or optimizer state. Architecture,
adapter shape and finite tensors are checked at loading, but the loader does
not hash all base weight files: the caller must supply the identical base checkpoint.

Python API:

```python
from ttt_frame.videoqa import VideoTTTConfig, VideoTTTMemory

memory = VideoTTTMemory(VideoTTTConfig(model_path=MODEL, local_files_only=True))
memory.ingest_video("day1.mp4")
memory.ingest_video("day2.mp4")
stats = memory.finish_ingest()
answer = memory.answer("Where was the mug last seen?")
memory.save("memories/home1")
memory.reset()  # required before the next household
```

### Evaluate on an existing MEOWBench release

The bridge remains in the sibling repository's
`meowbench/adapters/ttt_lact.py`; select **`--backend lora`** for generated answers.
The default backend is still the old CLIP/LaCT experiment. The new backend uses
the harness's standard question formatting and supports its multiple-choice,
numeric and open answer payloads. Open answers still need a judge and must not
be treated as zero before judging.

```bash
cd ~/meowbench  # the checkout whose remote is quzitsix/test_1
SUITE=/path/to/prepared/egocentric_suite
python scripts/run_ttt_pilot.py \
  --suite "$SUITE" --model-path "$MODEL" --local-files-only \
  --out runs/ttt_egovideo_pilot --limit 8 \
  --arms blind memory base-read \
  --chunk-seconds 30 --frames-per-chunk 4 \
  --steps-per-chunk 12 --teacher-max-new-tokens 512
```

This reuses `manifest.json`, `envs.jsonl` and `items.jsonl` as prepared by
MEOWBench. It does not download, mine or change annotations. It checks media
paths before loading the model, then the harness stages and revokes the videos.
Each arm sees the same selected items and chronological sessions:

| Arm | Ingestion | Query |
|---|---|---|
| `blind` | receives no video; no updates | original VLM |
| `memory` | video self-distillation into LoRA | question + LoRA parameters |
| `base-read` | same video/training budget as memory | LoRA disabled |
| `notes` | frozen VLM descriptions | question + retained text notes |
| `oracle` | sampled frames retained | question + explicit frames |

`notes` and `oracle` are frozen HF controls. They sample session midpoints while
TTT samples chunk starts; frame counts can differ for short final chunks. They
are diagnostic comparisons, not an identical-input ablation.

Use `--trace-teacher` to persist observations and pseudo-QA in a separate
**evaluator log**. The learner never reads this log; the saved memory remains
parameter-only. With tracing enabled, text derived from the video does remain
on disk outside the checkpoint, for auditing training-target quality.

The benchmark repository's `docs/TTT_EVALUATION.md` gives complete commands for
saved-memory/video controls, update-strength comparisons and standalone
Markdown/HTML reports (`scripts/report_ttt_evaluation.py`).

Outputs include per-arm predictions, protocol summary, score report, numeric
`ttt_metrics.jsonl`, and paired `comparison.json`. Errors stay in the score
denominator. The experiment directory must be new, so a pilot cannot silently
reuse predictions from a different model/configuration. The runner pins UTF-8
for subprocess pipes, including on Windows paths with Chinese characters.

For direct use of the harness:

```bash
meowbench run --suite "$SUITE" --context-mode memory \
  --run-id video_lora_first --limit 8 \
  --system "python -m meowbench.adapters.ttt_lact --backend lora --model-path $MODEL --local-files-only --context-mode memory"
```

`--max-chunks 0` (default) consumes the whole input. A positive value caps each
session to its first N sampled chunks and is **only a throughput smoke budget**;
it can exclude required evidence. `--limit` selects questions, not shorter video
history. Reduce frames, resolution, or selected environments first when memory
is tight; `--gradient-checkpointing` applies to the text training pass.

### What to measure next

Start with real first-person clips from the already prepared suite, preferably
an automatically scored subset. Inspect zero-frame sessions, malformed QA
(`caption_only_chunks`), truncation, errors and video revocation before reading
accuracy. Compare memory against blind and base-read; then add the existing
`hf_vlm --context-mode memory` notes baseline and the video oracle. Their current
frame-sampling and token budgets differ, so match budgets before making a method
comparison. Report accuracy/paired gain, refusal rate, ingestion time, query
latency and LoRA bytes. `ingest_peak_cuda_allocated_bytes` includes the base and
activations; `memory_bytes` counts only episodic LoRA tensors, excluding the
base, initial reference copy and temporary optimizer state.

The default 4 frames per 30 seconds is coarse and can miss brief handovers.
For the first human–object pilot, consider `--chunk-seconds 8 --frames-per-chunk 4`
and inspect perception before increasing the sequence length. These are ordered
RGB samples, not a full-frame-rate action-recognition pipeline.

A fall in training loss only shows target fitting. Useful recall, retention
across intervening chunks, and person–object binding after relocation require
separate held-out questions. This baseline does not yet establish any of those.
See the new dated entry in [WORK_REPORT.md](docs/WORK_REPORT.md) for actual checks.

---

## Original LaCT mechanism experiment

The memory is a LaCT layer — large-chunk test-time training, from
[*Test-Time Training Done Right*](https://arxiv.org/abs/2505.23884) (MIT
licensed; the untouched original is kept at `ttt_frame/_lact_upstream.py`). It is
pure PyTorch: no Triton, no CUDA extension, so the same code runs on a multi-GPU
Linux server and on a Windows laptop.

This repo deliberately contains **no benchmark harness**. It answers one
question — can fast weights hold a binding? — and it is meant to stay small
enough to read in one sitting.

**Picking this up cold?** Read [`docs/HANDOVER.md`](docs/HANDOVER.md) first: the
design rationale, the nine defects worth not rediscovering, and the five times a
plausible inference turned out to be wrong. [`docs/WORK_REPORT.md`](docs/WORK_REPORT.md)
is the narrative of what was run and what it showed, with each number marked by
whether it was verified first-hand. The sibling `meowbench` repo has its own pair
of these covering the benchmark side.

---

## Install

```bash
pip install -e ".[exp,dev]"
pytest -q                       # 14 tests, no GPU needed
```

On Windows, `torch.compile` shells out to MSVC and fails with
`RuntimeError: Compiler: cl is not found`, so compilation is off by default.
Enable it on Linux with `MEOW_TTT_COMPILE=1` for a modest speedup.

## The memory

```python
from ttt_frame.lact import LaCTMemory

mem = LaCTMemory(dim=512, head_dim=64, base_lr=0.05)

mem.write(objects, values=owners)   # bind object[i] -> owner[i]
mem.write(more_frames)              # self-associative when values is omitted
owner_guess = mem.read(objects)     # query after the video is gone
```

A SwiGLU fast weight `f_W(x) = W1(silu(W0 x) * (W2 x))` is updated by one
gradient step per chunk on `L = -f_W(k)ᵀv`, optionally with Muon
(Newton–Schulz) orthogonalisation, then renormalised so each row keeps its
pre-update norm. State size is fixed and independent of stream length.

`write(tokens, values=...)` is the part that matters for a binding experiment:
with identity projections `k` and `v` are otherwise the same vector, so the
memory could only ever return what it was queried with.

---

## A correctness fix worth knowing about

Upstream's `minimal_implementations/bidirectional_lact_layer.py` contains

```python
if use_muon:
    w0 = zeropower_via_newtonschulz5(dw0)   # assigns to w0, not dw0
    ...
w0 = w0 + dw0
```

which overwrites the fast weight with the orthogonalised *gradient* before
adding the gradient again, destroying the prior state — so the layer cannot
carry memory at all. `use_muon=True` is the default.

Verified by setting the learning rate to zero, where the update must be a no-op:

| | lr=0 is a no-op? | max deviation from `f_w(q)` |
|---|---|---|
| upstream, `use_muon=False` | yes | 0.000 |
| upstream, `use_muon=True` (default) | **no** | **20.416** |
| this repo, both settings | yes | 0.0003 |

Every other implementation in the same upstream repository (the causal layer,
`lact_ar_video/.../ar_lact_swa_repeat.py`, the Triton kernels) uses the `dw`
form, so this is a typo confined to that one minimal file rather than the
intended algorithm. Pinned by `test_zero_lr_is_a_noop`.

---

## The experiment

```bash
python scripts/exp_scene_change.py --trials 30 --fillers 8 16 24 32
```

Bindings are written in "home A", then N neutral filler sessions arrive, then
the bindings are queried. Two arms hold evidence distance fixed and vary only
the *visual context* of the filler:

- **SAME** — filler from the same home
- **MOVED** — filler from a different home (the move)

Perception is deliberately factored out: objects and people are large coloured
shapes that a frozen CLIP separates cleanly. An earlier video fixture rendered
its facts as *text in pixels*, which CLIP cannot read — every frame within a
session embedded at cosine 0.9996 of every other — so the arms came out
identical for a reason that had nothing to do with memory. Removing perception
from the loop is what makes a failure attributable.

### What it found

Controls first, because the result is meaningless without them: pure
associative recall is 1.00 across every learning rate and head configuration
tested; a *permuted* binding is recalled as the permutation and not as identity,
so the readout genuinely uses the memory; and a no-write baseline sits at chance.

**Embedding anisotropy dominates everything else.** Raw CLIP embeddings occupy a
narrow cone, so unrelated filler images sit at cosine ~0.90 to the object keys
and a filler write lands almost on top of them. Subtracting the corpus mean and
renormalising drops that to 0.646:

| filler writes | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| raw CLIP | 1.000 | 0.750 | 0.250 | 0.250 | 0.250 | 0.250 |
| mean-centred | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 |

(4 bindings, chance 0.250.) Without centring the memory looks catastrophically
forgetful after two writes; with it, retention runs to 8+ writes. The collapse
was a property of the *representation*, not of the update rule — reproduce it
with `--no-center`.

**The SAME/MOVED gap is not a real effect: its sign is set by an arbitrary
preprocessing choice.** Mean-centring needs a reference set, and nothing in the
problem specifies which. Four defensible choices, 40 trials each:

| centring reference | filler-A/obj | filler-B/obj | gap @8 | gap @16 |
|---|---|---|---|---|
| narrow (only embeddings in play) | 0.262 | 0.456 | **+0.037** | **−0.038** |
| broad (36 objects × 2 homes) | 0.145 | 0.316 | −0.019 | −0.181 |
| home-A only (what an online system could compute) | 0.529 | 0.157 | −0.062 | −0.281 |
| none | 0.861 | 0.850 | 0.000 | 0.000 |

The gap flips sign *between* references, and under the narrow reference it flips
*within* itself between 8 and 16 filler writes. An independent re-analysis using
the narrow reference reported SAME 0.775 vs MOVED 0.631 (gap **+0.144**) and we
reproduced those figures exactly; with the broad reference the same code gives
SAME 0.944 vs MOVED 0.963. Both are correct computations of different things.

What actually drives it is visible in the middle columns: centring changes how
much each filler stream overlaps the stored keys, and *that* is what predicts
retention. Under the home-A reference the overlap ordering reverses
(filler A 0.529 > filler B 0.157) and so does the outcome. There is no residual
"scene change" signal once key overlap is accounted for.

So: **this fixture cannot answer whether relocation damages bindings.** Stimulus
geometry dominates, and "the move" is not separable from it here.

### The interaction, and a cell that fakes chance

Two further reasons not to read the diagonal. Running the 2×2 (`--full`) at 8
filler writes, broad centring, `d` = distinct predictions out of 4:

| | query in A | query in B |
|---|---|---|
| **filler from A** | 0.925 ± 0.052  d=3.70 | 0.412 ± 0.054  d=1.65 |
| **filler from B** | 0.975 ± 0.034  d=3.90 | 0.975 ± 0.034  d=3.90 |

filler main effect +0.306 · query main effect −0.256 · **interaction +0.513**

Cue drift is *free* under filler B (0.975 → 0.975) and *catastrophic* under
filler A (0.925 → 0.412), so an averaged "cue drift costs 0.256" is meaningless.
And at 32 filler writes, (filler A, query B) returns `0.250 ± 0.000` with
`d=1.00`: the memory has saturated and returns **the same person for all four
objects**, every seed. That is numerically identical to guessing with zero
variance, so accuracy alone cannot distinguish collapse from chance. `--full`
now flags such cells instead of averaging them into an effect.

### What does survive

1. **Anisotropy, not the update rule, sets the retention horizon.** This is the
   largest and most robust effect in the experiment. Raw CLIP embeddings share
   ~92% of their energy with a single mean direction, so a "neutral" room image
   is a near-duplicate of every object key. The decisive control: random unit
   vectors rotate the fast weights by the *same* amount as room images
   (cos(W_bind, W_8) = 0.857 vs 0.880) yet retention stays at 1.000 out to 16
   writes, while room filler collapses by 2 writes. Rotation magnitude is not
   the mechanism; key aliasing is. Dose–response in key-space cosine is monotone
   and the real images land exactly on the curve.
2. **Damage requires the filler to act as a KEY.** `write(k=filler, v=zeros)`
   retains 1.000 at every filler count; `k=random, v=filler` retains 0.958 at 8.
   What the interference carries as a *value* is secondary.
3. **The bindings are not erased — the readout collapses.** The four returned
   vectors converge (pairwise cos 0.789 → 0.972) onto a shared direction while
   `cos(read, correct person)` barely moves. Subtracting the mean of the reads
   before argmax restores 1.000 at 2 writes (from 0.271); Hungarian assignment
   gives 1.000 out to 4 and 0.750 at 8. **The information survives writes that
   the shipped readout scores at chance.**
4. Muon is load-bearing in the opposite direction to expectation: with
   `use_muon=False` the update is ~470× smaller (|dW|/|W| = 0.00057 vs 0.269)
   and nothing is written at all — 0.250 even at zero fillers.
5. Learning rate (0.001–1.0) and capacity (head_dim 16–512) are both flat. No
   hyperparameter buys retention; only the representation does.

Results 1–3 reproduce across an RTX 4060 and an RTX 4090 and across two
independent implementations.

Treat these numbers as a mechanism probe on synthetic stimuli, not as a claim
about household video.

---

## Layout

```
ttt_frame/lact.py             the memory (write/read, fixed-size state)
ttt_frame/_lact_upstream.py   upstream original, for diffing
scripts/exp_scene_change.py   the binding-across-scene-change experiment
scripts/exp_centering_sensitivity.py
                              why the SAME/MOVED gap is not a finding
scripts/fetch_weights.sh      fetch the encoder on a machine without hub access
tests/test_lact.py            15 tests, including the upstream-bug regression
docs/HANDOVER.md              start here if you are new to the project
docs/WORK_REPORT.md           what was run, and what each number is worth
docs/TTT_PROTOTYPE.md         early diagnosis log; partly superseded by this README
```
