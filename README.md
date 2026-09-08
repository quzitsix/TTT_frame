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

**Bindings survive the move.** With centring on, 30 trials per cell:

| filler writes | SAME | MOVED | gap |
|---|---|---|---|
| 8 | 0.942 | 0.983 | −0.042 |
| 16 | 0.667 | 0.808 | −0.142 |
| 24 | 0.425 | 0.642 | −0.217 |
| 32 | 0.392 | 0.608 | −0.217 |

The gap is *negative*: moving house is consistently **easier** than staying put,
which is the opposite of the predicted failure. The likely mechanism is that
same-home filler shares its background with the stored keys and therefore
overlaps them more, so it overwrites more — i.e. what damages a fast-weight
binding is representational overlap with the intervening data, not context
change as such. Both arms decay with filler count, so this is interference
rather than a context effect.

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
