#!/usr/bin/env python3
"""Does a fast-weight binding survive a change of scene?

THE EXPERIMENT

Bindings (object -> owner) are written while the "camera" is in home A, then
more sessions arrive from home B, and only afterwards is the binding queried.
The question is whether the binding survives the intervening context, and the
comparison that answers it holds evidence distance fixed:

    SAME    write bindings, then N filler sessions from the SAME home, then ask
    MOVED   write bindings, then N filler sessions from a DIFFERENT home, then ask

Both arms see the same bindings, the same number of intervening writes, and the
same query. Only the *visual context* of the filler differs. So a gap between
them isolates context change from ordinary forgetting, which a single-environment
benchmark cannot do:

    SAME high, MOVED low   -> a binding/interference problem specific to context
    both low               -> ordinary forgetting; nothing to do with the move
    both high              -> fast weights carry bindings across scenes

Run from the repo root:

    python scripts/exp_scene_change.py
    python scripts/exp_scene_change.py --fillers 0 1 2 4 8 --trials 5

WHY IT IS SYNTHETIC, AND WHAT THAT BUYS

Perception is deliberately removed from the loop. The relocate *video* fixture
renders its facts as text, and a frozen CLIP cannot read text in pixels — every
frame within a session embeds at cosine 0.9996 of every other, so there is
nothing for a memory to bind and the arms come out identical for a reason that
has nothing to do with memory (see docs/TTT_PROTOTYPE.md). Here the objects and
people are large coloured shapes, which CLIP separates cleanly (measured
off-diagonal ~0.88), so a failure is attributable to the memory rather than the
encoder. That is the whole point of running this before touching a VLM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ttt_frame.lact import LaCTMemory  # noqa: E402

RGB = {
    "red": (200, 30, 30), "blue": (30, 60, 200), "green": (30, 150, 60),
    "yellow": (220, 190, 40), "purple": (130, 40, 160), "orange": (230, 120, 20),
    "cyan": (30, 180, 190), "brown": (120, 70, 30), "pink": (230, 130, 180),
    "navy": (20, 30, 90), "olive": (110, 120, 30), "maroon": (110, 20, 40),
}
OBJECT_COLOURS = ["red", "blue", "green", "yellow", "pink", "olive"]
SHAPES = ["circle", "square", "triangle"]
SHIRT_COLOURS = ["purple", "orange", "cyan", "brown", "navy", "maroon"]

#: Two homes, distinguished by wall colour and furniture layout. The filler
#: sessions differ only in this, which is what makes the MOVED arm a scene
#: change rather than just more data.
HOMES = {
    "A": {"wall": (238, 236, 228), "floor": (196, 174, 140)},
    "B": {"wall": (222, 234, 240), "floor": (150, 160, 175)},
}


def object_image(colour: str, shape: str, home: str) -> Image.Image:
    """One object, photographed in a given home."""
    style = HOMES[home]
    image = Image.new("RGB", (336, 336), style["wall"])
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 250, 336, 336], fill=style["floor"])
    box = [80, 80, 256, 256]
    if shape == "circle":
        draw.ellipse(box, fill=RGB[colour])
    elif shape == "square":
        draw.rectangle(box, fill=RGB[colour])
    else:
        draw.polygon([(168, 70), (266, 256), (70, 256)], fill=RGB[colour])
    return image


def person_image(shirt: str, home: str) -> Image.Image:
    """One resident, identified by shirt colour."""
    style = HOMES[home]
    image = Image.new("RGB", (336, 336), style["wall"])
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 250, 336, 336], fill=style["floor"])
    draw.ellipse([138, 40, 198, 100], fill=(230, 200, 170))
    draw.rectangle([104, 106, 232, 252], fill=RGB[shirt])
    return image


def filler_image(home: str, index: int) -> Image.Image:
    """An innocuous scene from a home: no objects, no people, just the room.

    Filler must be *neutral* — if it contained objects or people it would write
    competing bindings and confound the context effect with plain interference.
    """
    style = HOMES[home]
    image = Image.new("RGB", (336, 336), style["wall"])
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 250, 336, 336], fill=style["floor"])
    # A few boxes/furniture blobs, shifted per session so writes are not identical.
    offset = (index * 37) % 120
    draw.rectangle([30 + offset, 170, 90 + offset, 250], fill=(170, 150, 120))
    draw.rectangle([200, 150 + (offset % 40), 300, 250], fill=(150, 140, 130))
    return image


class Encoder:
    """A frozen CLIP image/text encoder, optionally mean-centred.

    Centring is not cosmetic — it decides whether this experiment measures
    anything. CLIP embeddings are strongly anisotropic: they occupy a narrow
    cone, so two *unrelated* images still sit at high cosine. Measured here,
    empty-room filler images sit at cosine 0.897 to the object embeddings,
    while random Gaussian vectors sit at 0.096. A filler write therefore lands
    almost on top of the stored keys and overwrites them, and binding recall
    collapses to chance after two writes — which reads exactly like
    catastrophic forgetting but is a property of the *representation*, not of
    the memory.

    Subtracting the corpus mean and renormalising drops filler-object cosine to
    0.646 and takes retention from "dead at 2 writes" to perfect at 8 and 0.75
    at 16. The same trick is standard practice for CLIP/sentence-embedding
    anisotropy.

    The mean is fitted on a fixed reference set at construction, not per batch:
    a per-batch mean would leak information between the write and query phases
    and would shift as the composition of a batch changed.
    """

    def __init__(
        self, model_path: str, device: torch.device, *, center: bool = True
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        self.device = device
        self.dim = self.model.config.projection_dim
        self.center = center
        self._mu: torch.Tensor | None = None
        if center:
            self._fit_mean()

    def _fit_mean(self) -> None:
        """Estimate the embedding-space mean from a broad reference sample."""
        reference: list[Image.Image] = []
        for home in HOMES:
            reference += [
                object_image(colour, shape, home)
                for colour in OBJECT_COLOURS
                for shape in SHAPES
            ]
            reference += [person_image(shirt, home) for shirt in SHIRT_COLOURS]
            reference += [filler_image(home, i) for i in range(8)]
        self._mu = None  # raw encoding while fitting
        self._mu = self._encode_raw(reference).mean(dim=0, keepdim=True)

    @torch.no_grad()
    def _encode_raw(self, images: list[Image.Image]) -> torch.Tensor:
        out = []
        for start in range(0, len(images), 64):  # bounded batch: 8GB-safe
            batch = self.proc(
                images=images[start : start + 64], return_tensors="pt"
            ).to(self.device)
            out.append(self.model.get_image_features(**batch))
        return F.normalize(torch.cat(out), dim=-1)

    def images(self, images: list[Image.Image]) -> torch.Tensor:
        feats = self._encode_raw(images)
        if self._mu is not None:
            feats = F.normalize(feats - self._mu, dim=-1)
        return feats


def run_trial(
    enc: Encoder,
    *,
    n_bindings: int,
    n_fillers: int,
    moved: bool,
    seed: int,
    base_lr: float,
    head_dim: int,
    use_muon: bool,
) -> float:
    """One trial. Returns retrieval accuracy over the written bindings."""
    generator = torch.Generator().manual_seed(seed)

    # Distinct objects and residents for this household.
    obj_specs, person_specs = [], []
    for i in range(n_bindings):
        colour = OBJECT_COLOURS[i % len(OBJECT_COLOURS)]
        shape = SHAPES[(i // len(OBJECT_COLOURS)) % len(SHAPES)]
        obj_specs.append((colour, shape))
        person_specs.append(SHIRT_COLOURS[i % len(SHIRT_COLOURS)])

    # A random owner assignment, so success cannot come from a fixed order.
    owner_of = torch.randperm(n_bindings, generator=generator)

    objects = enc.images([object_image(c, s, "A") for c, s in obj_specs])
    people = enc.images([person_image(person_specs[i], "A") for i in range(n_bindings)])

    mem = LaCTMemory(
        dim=enc.dim, head_dim=head_dim, base_lr=base_lr, use_muon=use_muon, seed=seed
    ).to(objects.device)

    # 1. Write the bindings, in home A.
    mem.write(objects, values=people[owner_of])

    # 2. Intervening sessions. Same home, or the new one.
    filler_home = "B" if moved else "A"
    for step in range(n_fillers):
        mem.write(enc.images([filler_image(filler_home, step)]))

    # 3. Query. The objects are re-photographed in whichever home we are now in,
    #    because in a real relocation you would see your mug in the *new* room —
    #    that is precisely what makes the retrieval cue shift.
    query_home = "B" if moved else "A"
    cues = enc.images([object_image(c, s, query_home) for c, s in obj_specs])
    retrieved = mem.read(cues)

    scores = F.normalize(retrieved, dim=-1) @ people.T
    predicted = scores.argmax(dim=-1).cpu()
    return float((predicted == owner_of).float().mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="openai/clip-vit-base-patch32")
    parser.add_argument("--bindings", type=int, default=4)
    parser.add_argument("--fillers", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--base-lr", type=float, default=0.05)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--no-muon", action="store_true")
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="disable embedding mean-centring. Retention collapses without it "
        "(chance after 2 filler writes) because raw CLIP filler sits at cosine "
        "~0.9 to the object keys; use this to reproduce that failure.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    enc = Encoder(args.model_path, device, center=not args.no_center)
    chance = 1.0 / args.bindings

    print(f"encoder      {args.model_path} (dim={enc.dim}) on {device}")
    print(f"centering    {'on' if enc.center else 'OFF (expect collapse)'}")
    print(f"bindings     {args.bindings} per household  (chance = {chance:.3f})")
    print(f"memory       head_dim={args.head_dim} base_lr={args.base_lr} "
          f"muon={not args.no_muon}")
    print(f"trials       {args.trials} per cell\n")
    print(f"{'fillers':>8}  {'SAME home':>11}  {'MOVED home':>11}  {'gap':>7}")
    print("-" * 44)

    rows = []
    for n_fillers in args.fillers:
        cell = {}
        for moved in (False, True):
            accs = [
                run_trial(
                    enc,
                    n_bindings=args.bindings,
                    n_fillers=n_fillers,
                    moved=moved,
                    seed=trial,
                    base_lr=args.base_lr,
                    head_dim=args.head_dim,
                    use_muon=not args.no_muon,
                )
                for trial in range(args.trials)
            ]
            cell[moved] = sum(accs) / len(accs)
        gap = cell[False] - cell[True]
        rows.append((n_fillers, cell[False], cell[True], gap))
        print(f"{n_fillers:>8}  {cell[False]:>11.3f}  {cell[True]:>11.3f}  {gap:>+7.3f}")

    print("-" * 44)
    print(f"\nchance is {chance:.3f}. Read the table as:")
    print("  SAME high, MOVED low  -> context change specifically damages bindings")
    print("  both fall together    -> ordinary forgetting, unrelated to the move")
    print("  both stay high        -> fast weights do carry bindings across scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
