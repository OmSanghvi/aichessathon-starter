"""Regression checks for 50-move-rule search state."""

import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.board import EMPTY, IS_SLIDER, N_OFFSETS, OFFSETS  # noqa: E402
from nengine.search import (  # noqa: E402
    MAX_PLY,
    PST,
    PST_KING_END,
    PST_KING_MID,
    ZOB_CASTLE,
    ZOB_EP,
    ZOB_PIECE,
    ZOB_SIDE,
    new_tt,
    next_halfmove_clock,
    search_root,
    zobrist,
)
from nengine.test_perft import to_arrays  # noqa: E402

NNUE_ACC = np.zeros((2, 512), dtype=np.int32)
NNUE_FT = np.zeros((768, 512), dtype=np.int16)
NNUE_OUT = np.zeros(1024, dtype=np.int16)


def main() -> None:
    assert next_halfmove_clock(2, EMPTY, 99) == 100  # quiet knight move
    assert next_halfmove_clock(1, EMPTY, 99) == 0  # pawn move
    assert next_halfmove_clock(5, 7, 99) == 0  # queen capture

    position = chess.Board("7k/8/8/8/8/8/8/K6R w - - 100 1")
    board, side, castling, ep = to_arrays(position)
    tt = new_tt()
    killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
    history = np.zeros((2, 120, 120), dtype=np.int32)
    counters = np.zeros(2, dtype=np.int64)
    game_hashes = np.zeros(600, dtype=np.int64)
    root_hash = zobrist(board, side, castling, ep, ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP)
    score, move = search_root(
        board, side, castling, ep, 4,
        OFFSETS, N_OFFSETS, IS_SLIDER, PST, PST_KING_MID, PST_KING_END,
        tt[0], tt[1], tt[2], tt[3], tt[4], killers, history, counters, 1_000_000,
        ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP, 0, game_hashes, 0, root_hash,
        position.halfmove_clock,
        NNUE_ACC, NNUE_FT, NNUE_OUT, 0, False,
    )
    print(f"50-move root: score {score}, move {move}")
    if score != 0 or move != 0:
        raise SystemExit("FAIL a 50-move draw was searched as a live position")
    print("50-move regression checks passed")


if __name__ == "__main__":
    main()
