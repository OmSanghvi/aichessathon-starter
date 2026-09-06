"""Regression checks for the mailbox static-exchange evaluator."""

import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.board import IS_SLIDER, N_OFFSETS, OFFSETS, SQ120, encode  # noqa: E402
from nengine.search import see  # noqa: E402
from nengine.test_perft import to_arrays  # noqa: E402


def exchange(fen: str, uci: str) -> tuple[int, bool]:
    position = chess.Board(fen)
    board, side, castling, ep = to_arrays(position)
    before = board.copy()
    move = chess.Move.from_uci(uci)
    encoded = encode(int(SQ120[move.from_square]), int(SQ120[move.to_square]), 0, 0)
    score = see(board, side, castling, ep, encoded, OFFSETS, N_OFFSETS, IS_SLIDER)
    return score, bool(np.array_equal(board, before))


def main() -> None:
    cases = [
        ("7k/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0, "equal pawn trade"),
        ("7k/8/2p5/3p4/4Q3/8/8/4K3 w - - 0 1", "e4d5", -800, "queen loses to pawn"),
        ("7k/8/2p5/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5", 800, "pawn wins queen"),
    ]
    for fen, move, expected, label in cases:
        actual, restored = exchange(fen, move)
        print(f"{label}: SEE {actual:+d}")
        if actual != expected or not restored:
            raise SystemExit(f"FAIL {label}: expected {expected:+d}, restored={restored}")
    print("SEE regression checks passed")


if __name__ == "__main__":
    main()
