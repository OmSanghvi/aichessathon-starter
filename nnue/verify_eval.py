"""Check the agent-side evaluator against the training net, and time it.

Three things must hold before this goes anywhere near agent.py:
  1. the numba kernel agrees with the numpy reference in export_weights
  2. both agree with the float torch net within quantisation error
  3. it is fast enough to be worth the depth it costs
"""

import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402

from nnue.export_weights import reference_forward  # noqa: E402
from nnue.features import features_from_board  # noqa: E402
from nnue.nnue_eval import NnueEvaluator  # noqa: E402
from nnue.train import CP_SCALE, NNUE  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4rrk1/pp1n1ppp/2pb4/3p4/3P4/2NBP3/PP3PPP/2R2RK1 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "8/1P6/8/8/8/2k5/8/6K1 w - - 0 1",
]


def main() -> None:
    weights = OUT / "nnue_weights.npz"
    ckpt = OUT / "net.pt"
    if not weights.exists() or not ckpt.exists():
        raise SystemExit("train and export first (nnue/train.py, nnue/export_weights.py)")

    evaluator = NnueEvaluator(weights)
    evaluator.warm()

    with np.load(str(weights)) as z:
        wdict = {k: z[k] for k in z.files}

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = NNUE(state["acc"], state["hidden"])
    model.load_state_dict(state["state"])
    model.eval()

    print(f"{'position':<44} {'numba':>8} {'numpy':>8} {'float':>8}")
    kernel_vs_numpy = []
    kernel_vs_float = []
    for fen in FENS:
        board = chess.Board(fen)
        pos = agent.board_to_sunfish(board)

        numba_cp = evaluator.evaluate_board_string(pos.board)

        idx = features_from_board(board)
        numpy_cp = reference_forward(wdict, idx.astype(np.int64))

        dense = torch.zeros(1, 768)
        dense[0, torch.from_numpy(idx.astype(np.int64))] = 1.0
        with torch.no_grad():
            float_cp = float(model(dense)[0, 0]) * CP_SCALE

        kernel_vs_numpy.append(abs(numba_cp - numpy_cp))
        kernel_vs_float.append(abs(numba_cp - float_cp))
        print(f"{fen[:44]:<44} {numba_cp:>8d} {numpy_cp:>8d} {float_cp:>8.0f}")

    print(
        f"\nnumba vs numpy reference: max diff {max(kernel_vs_numpy)} cp "
        f"(must be 0 - same integer maths)"
    )
    print(f"numba vs float net      : max diff {max(kernel_vs_float):.0f} cp (quantisation)")

    # Speed, on real positions.
    boards = [agent.board_to_sunfish(chess.Board(f)).board for f in FENS]
    reps = 4000
    t0 = time.perf_counter()
    for i in range(reps):
        evaluator.evaluate_board_string(boards[i % len(boards)])
    per_us = (time.perf_counter() - t0) / reps * 1e6
    print(f"\nspeed: {per_us:.2f} us/eval  ({1e6 / per_us:,.0f} evals/sec)")
    print(f"  engine node cost is ~30.5 us, so this adds {per_us / 30.5 * 100:.0f}% per node")

    if max(kernel_vs_numpy) != 0:
        raise SystemExit("FAIL: numba kernel disagrees with the numpy reference")
    print("\nOK")


if __name__ == "__main__":
    main()
