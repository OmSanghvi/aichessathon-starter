"""Quantise the trained net to integers and verify the integer path.

The agent's forward pass is numba-jitted int32 arithmetic, which is what makes an
evaluation cost ~5us instead of ~13us in numpy floats.

SCALES, spelled out, because getting these wrong is silent - the net simply plays
differently than it trained, and in v1 that produced a 645 cp mean error that only
showed up because this file checks:

  training clamps every activation to [0, 1]; we represent that range as 0..ACT_MAX

  ft_w = w * FT_SCALE,  ft_b = b * FT_SCALE      with FT_SCALE == ACT_MAX
    => acc = FT_SCALE * pre-activation
    => h0 = clamp(acc, 0, ACT_MAX)
       NOT acc / FT_SCALE. Dividing collapses activations to 0 or 1 and throws the
       entire range away. That was bug one.

  l1_w = w * W_SCALE,  l1_b = b * W_SCALE * ACT_MAX
    => h0 @ l1_w + l1_b = W_SCALE * ACT_MAX * pre-activation
    => shift right by log2(W_SCALE) to return to ACT_MAX units.
       SHIFT is log2(W_SCALE), not log2(ACT_MAX). That was bug two.

Overflow headroom: 32 features x 127 x 127 is about 5e5, far under 2^31.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nnue2.features import NUM_FEATURES  # noqa: E402
from nnue2.train import K_CP, NNUE  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

ACT_MAX = 127
FT_SCALE = 127.0
W_SCALE = 64.0
SHIFT = 6  # log2(W_SCALE)


def quantise(model: NNUE) -> dict[str, np.ndarray]:
    s = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    return {
        # Feature transformer stored feature-major so the agent can add whole rows
        # for the pieces on the board - the operation that runs ~32 times per eval.
        "ft_w": np.rint(s["ft.weight"].T * FT_SCALE).astype(np.int32),
        "ft_b": np.rint(s["ft.bias"] * FT_SCALE).astype(np.int32),
        "l1_w": np.rint(s["l1.weight"].T * W_SCALE).astype(np.int32),
        "l1_b": np.rint(s["l1.bias"] * W_SCALE * ACT_MAX).astype(np.int32),
        "l2_w": np.rint(s["l2.weight"].T * W_SCALE).astype(np.int32),
        "l2_b": np.rint(s["l2.bias"] * W_SCALE * ACT_MAX).astype(np.int32),
    }


def reference_forward(w: dict[str, np.ndarray], idx: np.ndarray) -> int:
    """Integer forward pass in numpy, mirroring the agent's numba kernel exactly."""
    acc = w["ft_b"].astype(np.int64).copy()
    for f in idx:
        acc += w["ft_w"][f]
    h0 = np.clip(acc, 0, ACT_MAX)  # already in ACT_MAX units; saturate only
    h1 = np.clip((h0 @ w["l1_w"].astype(np.int64) + w["l1_b"]) >> SHIFT, 0, ACT_MAX)
    out = int(h1 @ w["l2_w"].astype(np.int64).ravel()) + int(w["l2_b"][0])
    return round(out / (W_SCALE * ACT_MAX) * K_CP)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(OUT / "net.pt"))
    p.add_argument("--out", default=str(OUT / "nnue_weights.npz"))
    p.add_argument("--check", type=int, default=400)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NNUE(ck["acc"], ck["hidden"], dropout=0.0)
    model.load_state_dict(ck["state"])
    model.eval()

    w = quantise(model)
    # The agent must know whether the net outputs an absolute evaluation or a
    # correction to add to pos.score. Getting this wrong silently doubles or
    # discards the piece-square term, so it travels with the weights.
    w["residual"] = np.array([1 if ck.get("residual") else 0], dtype=np.int32)
    np.savez_compressed(args.out, **w)
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out} ({size:,} bytes, acc={ck['acc']}, hidden={ck['hidden']})")
    print(f"mode: {'RESIDUAL (add to pos.score)' if ck.get('residual') else 'absolute'}")
    print(f"checkpoint reported val_MAE {ck.get('val_mae_cp', float('nan')):.1f} cp")

    rng = np.random.default_rng(0)
    errs = []
    for _ in range(args.check):
        k = int(rng.integers(6, 32))
        idx = rng.choice(NUM_FEATURES, size=k, replace=False).astype(np.int64)
        dense = torch.zeros(1, NUM_FEATURES)
        dense[0, torch.from_numpy(idx)] = 1.0
        with torch.no_grad():
            float_cp = float(model(dense)[0, 0])
        errs.append(abs(float_cp - reference_forward(w, idx)))
    e = np.array(errs)
    print(f"quantisation error vs float net: mean {e.mean():.1f} cp  "
          f"p95 {np.percentile(e, 95):.1f} cp  max {e.max():.1f} cp")
    if e.mean() > 25:
        print("WARNING: large quantisation error - the integer path will play "
              "differently from what you trained. Do not ship this.")
    else:
        print("OK: integer path tracks the float net")
    print(f"\nTo ship: cp {args.out} weights/nnue.npz   (and add the agent kernel)")
    print("Nothing reaches the submission until you do that.")


if __name__ == "__main__":
    main()
