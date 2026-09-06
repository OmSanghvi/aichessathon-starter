"""Prove the agent's NNUE kernel matches the training reference, on real positions.

The agent computes features from sunfish's board string; training computes them
from a python-chess board. They must produce the same number for the same
position, or the net is being fed inputs it never trained on and its evaluation is
noise. This is the exact class of bug that made v1 worthless, so it is checked
directly rather than assumed.

Compares three things per position:
  agent kernel (_NNUE.evaluate on the sunfish board string)
  training reference_forward (integer path from python-chess features)
  float net (torch)
The first two must be IDENTICAL; the third within quantisation error.
"""

import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
from nnue2.export import reference_forward  # noqa: E402
from nnue2.features import NUM_FEATURES, features_from_board  # noqa: E402
from nnue2.train import NNUE  # noqa: E402

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "8/6pk/2p4p/8/1bp1pQ2/2n4P/2q2PPK/2B5 b - - 1 35",
    "6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42",
    "4rrk1/pp1n1ppp/2pb4/3p4/3P4/2NBP3/PP3PPP/2R2RK1 w - - 0 1",
    "8/1P6/8/8/8/2k5/8/6K1 w - - 0 1",
    "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7",
    "8/8/8/3k4/8/8/3r4/3K4 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
]


def main() -> None:
    if agent._NNUE is None:
        raise SystemExit("FAIL: agent did not load the net (weights/nnue.npz missing?)")

    ck = torch.load(ROOT / "nnue2" / "out" / "net.pt", map_location="cpu", weights_only=False)
    model = NNUE(ck["acc"], ck["hidden"], dropout=0.0)
    model.load_state_dict(ck["state"])
    model.eval()
    with np.load(ROOT / "nnue2" / "out" / "nnue_weights.npz") as z:
        w = {k: z[k] for k in z.files}

    print(f"{'position':<46}{'agent':>8}{'ref':>8}{'float':>8}")
    kernel_mismatches = 0
    float_gap = []
    for fen in FENS:
        board = chess.Board(fen)
        pos = agent.board_to_sunfish(board)
        agent_cp = agent._NNUE.evaluate(pos.board)

        idx = features_from_board(board).astype(np.int64)
        ref_cp = reference_forward(w, idx)

        dense = torch.zeros(1, NUM_FEATURES)
        dense[0, torch.from_numpy(idx)] = 1.0
        with torch.no_grad():
            float_cp = float(model(dense)[0, 0])

        flag = "" if agent_cp == ref_cp else "  <-- KERNEL MISMATCH"
        if agent_cp != ref_cp:
            kernel_mismatches += 1
        float_gap.append(abs(agent_cp - float_cp))
        print(f"{fen[:46]:<46}{agent_cp:>8}{ref_cp:>8}{float_cp:>8.0f}{flag}")

    print(f"\nagent kernel vs training reference: {kernel_mismatches} mismatches "
          f"(must be 0 - same integer maths, same features)")
    print(f"agent kernel vs float net: max {max(float_gap):.0f} cp (quantisation)")
    if kernel_mismatches:
        raise SystemExit("FAIL: the agent computes different features than training")
    print("\nOK: agent kernel is bit-identical to the training reference")


if __name__ == "__main__":
    main()
