"""How expensive is a FULL accumulator refresh?

This is the number that decides the integration. Sunfish's Position is an
immutable namedtuple: `move()` builds a new one, there is no make/unmake, so
there is no cheap place to thread an incrementally-updated accumulator. That
means every evaluation pays a full refresh: sum the accumulator columns of every
piece on the board, then run the head.

Incremental (~4 feature deltas) is the best case and needs a mutable board.
Full refresh (~32 pieces) is what we would actually pay with sunfish as-is.
"""

import time

import numpy as np
from numba import njit

IN, ACC, HID = 768, 256, 32
rng = np.random.default_rng(0)
W0 = rng.integers(-64, 64, size=(IN, ACC), dtype=np.int32)
W1 = rng.integers(-64, 64, size=(ACC, HID), dtype=np.int32)
W2 = rng.integers(-64, 64, size=(HID,), dtype=np.int32)
BIAS = rng.integers(-100, 100, size=ACC, dtype=np.int32)


@njit(cache=False)
def full_refresh(features, n_features, w0, w1, w2, bias):
    """Rebuild the accumulator from scratch, then run the head."""
    acc_n = bias.shape[0]
    acc = np.empty(acc_n, dtype=np.int32)
    for i in range(acc_n):
        acc[i] = bias[i]
    for k in range(n_features):
        f = features[k]
        for i in range(acc_n):
            acc[i] += w0[f, i]
    # clipped relu
    h0 = np.empty(acc_n, dtype=np.int32)
    for i in range(acc_n):
        v = acc[i] >> 6
        if v < 0:
            v = 0
        elif v > 127:
            v = 127
        h0[i] = v
    m = w1.shape[1]
    out = 0
    for j in range(m):
        s = 0
        for i in range(acc_n):
            s += h0[i] * w1[i, j]
        s >>= 6
        if s < 0:
            s = 0
        elif s > 127:
            s = 127
        out += s * w2[j]
    return out


@njit(cache=False)
def head_only(acc, w1, w2):
    """Just the head, for reference: the cost if the accumulator were free."""
    acc_n = acc.shape[0]
    h0 = np.empty(acc_n, dtype=np.int32)
    for i in range(acc_n):
        v = acc[i] >> 6
        if v < 0:
            v = 0
        elif v > 127:
            v = 127
        h0[i] = v
    m = w1.shape[1]
    out = 0
    for j in range(m):
        s = 0
        for i in range(acc_n):
            s += h0[i] * w1[i, j]
        s >>= 6
        if s < 0:
            s = 0
        elif s > 127:
            s = 127
        out += s * w2[j]
    return out


def bench(label, fn, reps, *args):
    fn(*args)  # compile
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    us = (time.perf_counter() - t0) / reps * 1e6
    print(f"{label:42s} {us:7.2f} us/eval  ({1e6/us:>10,.0f} evals/sec)")
    return us


def main() -> None:
    per_node_us = 30.49  # measured in bench_speed.py

    print("Smaller accumulator sizes trade evaluation quality for speed.\n")
    results = {}
    for acc_size in (256, 128, 64):
        w0 = rng.integers(-64, 64, size=(IN, acc_size), dtype=np.int32)
        w1 = rng.integers(-64, 64, size=(acc_size, HID), dtype=np.int32)
        bias = rng.integers(-100, 100, size=acc_size, dtype=np.int32)
        feats = np.array([i * 17 % IN for i in range(32)], dtype=np.int64)
        us = bench(
            f"full refresh 32 pieces, 768->{acc_size}->32->1",
            full_refresh,
            2000,
            feats,
            32,
            w0,
            w1,
            W2,
            bias,
        )
        results[acc_size] = us

    print()
    acc = rng.integers(-1000, 1000, size=256, dtype=np.int32)
    w1_256 = rng.integers(-64, 64, size=(256, HID), dtype=np.int32)
    bench("head only (incremental best case), 256", head_only, 3000, acc, w1_256, W2)

    print(f"\ncurrent engine per-node cost: {per_node_us:.2f} us")
    print("added cost as a fraction of a node, and the resulting slowdown:\n")
    for acc_size, us in results.items():
        frac = us / per_node_us
        slow = 1 + frac
        print(
            f"  acc={acc_size:4d}: +{us:6.2f} us = {frac*100:5.1f}% of a node "
            f"-> ~{slow:.2f}x slower search ({100/slow:.0f}% of current nodes)"
        )


if __name__ == "__main__":
    main()
