"""Quantise the trained net to integers and export it for the agent.

Why integers: the agent's forward pass is numba-jitted int32 arithmetic, which is
what makes an evaluation cost ~4.5us instead of ~13us in numpy floats. Floats
would also make the accumulator refresh slower for no accuracy that matters at
this net size.

Scheme. Activations are clipped to [0, 1] in training, so each layer's inputs map
onto a fixed integer range and we can use plain power-of-two shifts instead of
per-layer float scales:

    feature transformer : weights x FT_SCALE, accumulator in int32
    hidden / output      : weights x W_SCALE,  shift right by SHIFT after each

The export writes a single compressed .npz. It is small (a few hundred KB at
acc=256), loads instantly, and sits far inside both the 50MB zip cap and the 60s
import budget.

Run: .venv/bin/python nnue/export_weights.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnue.features import NUM_FEATURES  # noqa: E402
from nnue.train import CP_SCALE, NNUE  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

# Training clamps every activation to [0, 1]; we represent that range as 0..127,
# so one "1.0" of activation is ACT_MAX integer units.
#
# The scales follow from that, and getting them wrong is silent - the net simply
# plays differently than it trained - so the derivation is spelled out:
#
#   ft_w = w * FT_SCALE, ft_b = b * FT_SCALE
#     => acc = FT_SCALE * (pre-activation)
#     => quantised activation h0 = clamp(acc, 0, ACT_MAX), with FT_SCALE == ACT_MAX.
#        h0 is NOT divided by FT_SCALE: dividing collapses it to 0/1 and throws
#        away the whole activation range.
#
#   l1_w = w * W_SCALE, l1_b = b * W_SCALE * ACT_MAX
#     => h0 @ l1_w + l1_b = W_SCALE * ACT_MAX * (pre-activation)
#     => shift right by log2(W_SCALE) to land back in ACT_MAX units.
#
# Overflow headroom: 32 features x 127 x 127 is ~5e5, far under 2^31.
ACT_MAX = 127
FT_SCALE = 127.0  # feature-transformer weights and bias; equals ACT_MAX by design
W_SCALE = 64.0  # hidden/output weights
SHIFT = 6  # log2(W_SCALE), NOT log2(ACT_MAX)


def quantise(model: NNUE) -> dict[str, np.ndarray]:
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}

    # Feature transformer: stored transposed (feature-major) so the agent can add
    # whole rows for the pieces on the board, which is the operation that runs
    # 32 times per evaluation.
    ft_w = np.rint(state["ft.weight"].T * FT_SCALE).astype(np.int32)
    ft_b = np.rint(state["ft.bias"] * FT_SCALE).astype(np.int32)

    l1_w = np.rint(state["l1.weight"].T * W_SCALE).astype(np.int32)
    l1_b = np.rint(state["l1.bias"] * W_SCALE * ACT_MAX).astype(np.int32)

    l2_w = np.rint(state["l2.weight"].T * W_SCALE).astype(np.int32)
    l2_b = np.rint(state["l2.bias"] * W_SCALE * ACT_MAX).astype(np.int32)

    return {
        "ft_w": ft_w,
        "ft_b": ft_b,
        "l1_w": l1_w,
        "l1_b": l1_b,
        "l2_w": l2_w,
        "l2_b": l2_b,
    }


def reference_forward(weights: dict[str, np.ndarray], indices: np.ndarray) -> int:
    """Integer forward pass in numpy, mirroring what the agent will do in numba.

    Kept here so export can verify the quantised net against the float net; if
    these two disagree the net would silently play differently than it trained.
    """
    acc = weights["ft_b"].astype(np.int64).copy()
    for f in indices:
        acc += weights["ft_w"][f]
    # acc is already in ACT_MAX units (FT_SCALE == ACT_MAX), so just saturate.
    h0 = np.clip(acc, 0, ACT_MAX)

    h1 = h0 @ weights["l1_w"].astype(np.int64) + weights["l1_b"]
    h1 = np.clip(h1 >> SHIFT, 0, ACT_MAX)

    out = int(h1 @ weights["l2_w"].astype(np.int64).ravel()) + int(weights["l2_b"][0])
    # Undo the weight scaling and the centipawn scaling used in training.
    return int(round(out / (W_SCALE * ACT_MAX) * CP_SCALE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantise and export the net.")
    parser.add_argument("--checkpoint", default=str(OUT_DIR / "net.pt"))
    parser.add_argument("--out", default=str(OUT_DIR / "nnue_weights.npz"))
    parser.add_argument("--check", type=int, default=200, help="random positions to verify")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NNUE(ckpt["acc"], ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    model.eval()

    weights = quantise(model)
    np.savez_compressed(args.out, **weights)
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out} ({size:,} bytes, acc={ckpt['acc']}, hidden={ckpt['hidden']})")

    # Verify the quantised integer path tracks the float net.
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(args.check):
        n_pieces = int(rng.integers(6, 32))
        idx = rng.choice(NUM_FEATURES, size=n_pieces, replace=False).astype(np.int64)
        dense = torch.zeros(1, NUM_FEATURES)
        dense[0, torch.from_numpy(idx)] = 1.0
        with torch.no_grad():
            float_cp = float(model(dense)[0, 0]) * CP_SCALE
        int_cp = reference_forward(weights, idx)
        errors.append(abs(float_cp - int_cp))
    errors_arr = np.array(errors)
    print(
        f"quantisation error vs float net: mean={errors_arr.mean():.1f} cp  "
        f"p95={np.percentile(errors_arr, 95):.1f} cp  max={errors_arr.max():.1f} cp"
    )
    if errors_arr.mean() > 25:
        print("WARNING: quantisation error is large; the int path may play differently")


if __name__ == "__main__":
    main()
