"""Regression for the rated 13.Bxe7 positional exchange error."""

import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.search import (  # noqa: E402
    PST,
    PST_KING_END,
    PST_KING_MID,
    evaluate,
)
from nengine.test_perft import to_arrays  # noqa: E402

FEN = "r1bqk2r/1p2n1bp/p2p2p1/3PppB1/8/2NB4/PPP2PPP/R2Q1RK1 w kq - 2 13"


def white_score(board: chess.Board) -> int:
    arr, side, _castling, _ep = to_arrays(board)
    value = evaluate(arr, side, PST, PST_KING_MID, PST_KING_END)
    return value if side == 0 else -value


def main() -> None:
    before = chess.Board(FEN)
    exchanged = before.copy()
    exchanged.push_san("Bxe7")
    exchanged.push_san("Qxe7")
    re1 = before.copy()
    re1.push_san("Re1")

    before_score = white_score(before)
    exchange_score = white_score(exchanged)
    re1_score = white_score(re1)
    print(f"before {before_score:+d}; Bxe7 Qxe7 {exchange_score:+d}; Re1 {re1_score:+d}")
    if before_score - exchange_score < 60:
        raise SystemExit("FAIL bishop-pair loss is not visible to static evaluation")
    if re1_score <= exchange_score:
        raise SystemExit("FAIL quiet pressure is not preferred to the release trade")
    print("bishop-pair regression passed")


if __name__ == "__main__":
    main()
