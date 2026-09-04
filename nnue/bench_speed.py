"""Feasibility spike for an NNUE-style evaluation.

Two questions:
  1. How many nodes/sec does the current engine search? That sets how much time
     one evaluation may cost before the net becomes the bottleneck.
  2. How fast can we actually run a small net forward pass on one core, with
     numpy, with numba, and with onnxruntime?

Run: .venv/bin/python nnue/bench_speed.py
"""

import time

import numpy as np

# --------------------------------------------------------------------------
# 1. Current engine throughput
# --------------------------------------------------------------------------


def engine_nodes_per_sec() -> float:
    import sys

    sys.path.insert(0, ".")
    import agent

    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    ]
    total_nodes = 0
    total_time = 0.0
    for fen in positions:
        import chess

        pos = agent.board_to_sunfish(chess.Board(fen))
        s = agent.Searcher()
        s.soft = s.deadline = time.time() + 1.0
        t0 = time.time()
        try:
            for _ in s.search([pos]):
                pass
        except agent.Stop:
            pass
        total_time += time.time() - t0
        total_nodes += s.nodes
    return total_nodes / total_time


# --------------------------------------------------------------------------
# 2. Candidate net: HalfKP-lite feature transformer + small head
# --------------------------------------------------------------------------
# Shape: 768 inputs -> 256 accumulator -> 32 -> 1. The accumulator is updated
# INCREMENTALLY (add/subtract the columns of the features that changed), which
# is the whole point of NNUE: a move touches ~4 features, not all 768.

IN, ACC, HID = 768, 256, 32
rng = np.random.default_rng(0)
W0 = rng.integers(-64, 64, size=(IN, ACC), dtype=np.int16)
W1 = rng.integers(-64, 64, size=(ACC, HID), dtype=np.int16)
W2 = rng.integers(-64, 64, size=(HID,), dtype=np.int16)
acc0 = rng.integers(-1000, 1000, size=ACC, dtype=np.int32)


def numpy_incremental(reps: int) -> float:
    """Incremental accumulator update + head, pure numpy int arithmetic."""
    acc = acc0.copy()
    changed = np.array([12, 300, 455, 700], dtype=np.int64)  # ~4 features per move
    t0 = time.perf_counter()
    for _ in range(reps):
        # accumulator: add columns for added features, subtract for removed
        acc_local = acc + W0[changed[0]] + W0[changed[1]] - W0[changed[2]] - W0[changed[3]]
        # clipped relu
        h0 = np.clip(acc_local >> 6, 0, 127).astype(np.int16)
        # hidden layer
        h1 = np.clip((h0 @ W1) >> 6, 0, 127).astype(np.int16)
        # output
        _ = int(h1 @ W2)
    return (time.perf_counter() - t0) / reps * 1e6  # microseconds per eval


def numba_incremental(reps: int) -> float | None:
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=False, fastmath=False)
    def fwd(acc, w0, w1, w2, c0, c1, c2, c3):
        n = acc.shape[0]
        h0 = np.empty(n, dtype=np.int32)
        for i in range(n):
            v = acc[i] + w0[c0, i] + w0[c1, i] - w0[c2, i] - w0[c3, i]
            v >>= 6
            if v < 0:
                v = 0
            elif v > 127:
                v = 127
            h0[i] = v
        m = w1.shape[1]
        out = 0
        for j in range(m):
            s = 0
            for i in range(n):
                s += h0[i] * w1[i, j]
            s >>= 6
            if s < 0:
                s = 0
            elif s > 127:
                s = 127
            out += s * w2[j]
        return out

    w0 = W0.astype(np.int32)
    w1 = W1.astype(np.int32)
    w2 = W2.astype(np.int32)
    acc = acc0.astype(np.int32)
    fwd(acc, w0, w1, w2, 12, 300, 455, 700)  # warm/compile
    t0 = time.perf_counter()
    for _ in range(reps):
        fwd(acc, w0, w1, w2, 12, 300, 455, 700)
    return (time.perf_counter() - t0) / reps * 1e6


def onnx_dense(reps: int) -> float | None:
    """A dense (non-incremental) forward pass through onnxruntime, for contrast."""
    try:
        import onnxruntime as ort
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    torch.set_num_threads(1)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(IN, ACC)
            self.b = nn.Linear(ACC, HID)
            self.c = nn.Linear(HID, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = torch.clamp(self.a(x), 0, 1)
            x = torch.clamp(self.b(x), 0, 1)
            return self.c(x)

    net = Net().eval()
    path = "/tmp/_spike_net.onnx"
    dummy = torch.zeros(1, IN)
    torch.onnx.export(net, dummy, path, input_names=["x"], output_names=["y"])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    x = np.zeros((1, IN), dtype=np.float32)
    sess.run(None, {"x": x})  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {"x": x})
    return (time.perf_counter() - t0) / reps * 1e6


def main() -> None:
    nps = engine_nodes_per_sec()
    per_node_us = 1e6 / nps
    print(f"current engine: {nps:,.0f} nodes/sec  ({per_node_us:.2f} us per node)")
    print("  -> a net eval must cost well under this to not dominate the search\n")

    n = numpy_incremental(3000)
    print(f"numpy  incremental 768->256->32->1: {n:8.2f} us/eval  ({1e6/n:>10,.0f} evals/sec)")

    nb = numba_incremental(3000)
    if nb is None:
        print("numba: not available")
    else:
        print(f"numba  incremental 768->256->32->1: {nb:8.2f} us/eval  ({1e6/nb:>10,.0f} evals/sec)")

    ox = onnx_dense(1000)
    if ox is None:
        print("onnxruntime: not available")
    else:
        print(f"onnx   dense       768->256->32->1: {ox:8.2f} us/eval  ({1e6/ox:>10,.0f} evals/sec)")

    print("\nverdict inputs:")
    print(f"  engine budget per node : {per_node_us:.2f} us")
    for label, val in (("numpy", n), ("numba", nb), ("onnx", ox)):
        if val is not None:
            ratio = val / per_node_us
            print(f"  {label:6s}: {val:7.2f} us = {ratio:6.1f}x the current per-node cost")


if __name__ == "__main__":
    main()
