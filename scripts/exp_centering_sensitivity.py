#!/usr/bin/env python3
"""Does the SAME/MOVED gap depend on how the embeddings were centred?

It does, and that is the point: the gap's SIGN is decided by a preprocessing
choice the problem does not specify.

Mean-centring fixes CLIP's anisotropy (see the README), but "the mean" needs a
reference set, and several are defensible: only the embeddings in play, a wider
sample of the same stimulus space, or just the ones observed before the move —
the only option an online system could actually compute. Each yields a
different overlap between the filler streams and the stored keys, and retention
tracks that overlap rather than anything about relocation.

Two independent implementations of this experiment disagreed on the sign of the
gap (+0.144 vs -0.019 at 8 filler writes) purely because one used the narrow
reference and the other the broad one. Both computations were correct. That is
what this script exists to make visible, so nobody reports the gap as a finding
again.

    python scripts/exp_centering_sensitivity.py
"""
import sys
import torch, torch.nn.functional as F
sys.path.insert(0, 'scripts')
from exp_scene_change import (Encoder, object_image, person_image, filler_image,
                              OBJECT_COLOURS, SHAPES, SHIRT_COLOURS)
from ttt_frame.lact import LaCTMemory

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw = Encoder('openai/clip-vit-base-patch32', dev, center=False)
specs = [(OBJECT_COLOURS[i % 6], SHAPES[0]) for i in range(4)]
oA = raw.images([object_image(c, s, 'A') for c, s in specs])
oB = raw.images([object_image(c, s, 'B') for c, s in specs])
pA = raw.images([person_image(SHIRT_COLOURS[i], 'A') for i in range(4)])
pB = raw.images([person_image(SHIRT_COLOURS[i], 'B') for i in range(4)])
fA = raw.images([filler_image('A', i) for i in range(24)])
fB = raw.images([filler_image('B', i) for i in range(24)])

REFS = {}
# narrow: exactly what is in play (the workflow's choice)
REFS['narrow (in-play only)'] = torch.cat([oA, oB, pA, pB, fA, fB])
# broad: a wider sample of the same stimulus space (what TTT_frame ships)
broad = []
for home in ('A', 'B'):
    broad += [object_image(c, s, home) for c in OBJECT_COLOURS for s in SHAPES]
    broad += [person_image(s, home) for s in SHIRT_COLOURS]
    broad += [filler_image(home, i) for i in range(8)]
REFS['broad (36 obj x2 homes)'] = raw.images(broad)
# home-A only: the mean an online system could actually compute before moving
REFS['home-A only'] = torch.cat([oA, pA, fA])
# no centering at all
REFS['none'] = None

def table(ref, tag):
    mu = None if ref is None else ref.mean(0, keepdim=True)
    c = (lambda x: x) if mu is None else (lambda x: F.normalize(x - mu, dim=-1))
    O, P, FA, FB, QA, QB = c(oA), c(pA), c(fA), c(fB), c(oA), c(oB)
    ka = float((FA @ O.T).abs().mean()); kb = float((FB @ O.T).abs().mean())
    def trial(nf, fil, q, t):
        g = torch.Generator().manual_seed(t); own = torch.randperm(4, generator=g)
        mem = LaCTMemory(dim=512, head_dim=64, base_lr=0.05, seed=t).to(dev)
        mem.write(O, values=P[own])
        for s in range(nf):
            mem.write(fil[s % fil.shape[0]].unsqueeze(0))
        return float((F.normalize(mem.read(q), dim=-1) @ P.T).argmax(-1).cpu().eq(own).float().mean())
    N = 40
    out = []
    for nf in (8, 16):
        s = sum(trial(nf, FA, QA, t) for t in range(N)) / N
        m = sum(trial(nf, FB, QB, t) for t in range(N)) / N
        out.append((nf, s, m, s - m))
    print(f'{tag:26s} fA/obj={ka:.3f} fB/obj={kb:.3f}')
    for nf, s, m, g in out:
        print(f'    nf={nf:<3} SAME={s:.3f} MOVED={m:.3f} gap={g:+.3f}')

for tag, ref in REFS.items():
    table(ref, tag)
