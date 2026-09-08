# TTT_frame

A test-time-training (TTT) fast-weight memory, and experiments on **what it can
actually retain**.

The memory is a LaCT layer — large-chunk test-time training, from
[*Test-Time Training Done Right*](https://arxiv.org/abs/2505.23884) (MIT
licensed; the untouched original is kept at `ttt_frame/_lact_upstream.py`). It is
pure PyTorch: no Triton, no CUDA extension, so the same code runs on a multi-GPU
Linux server and on a Windows laptop.

This repo deliberately contains **no benchmark harness**. It answers one
question — can fast weights hold a binding? — and it is meant to stay small
enough to read in one sitting.

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
tests/test_lact.py            14 tests, including the upstream-bug regression
docs/TTT_PROTOTYPE.md         full diagnosis log
```
