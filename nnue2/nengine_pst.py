"""Exact current-Nengine static-evaluation anchor for residual NNUE training.

Residual training is only valid if its anchor is *the same function* used by the
search. Reimplementing material/PST tables here drifted when the engine moved to
PeSTO plus mobility, passed pawns and king safety. Build a mailbox and invoke the
real jitted evaluator instead, so a future evaluator change cannot silently train
an incompatible residual model.
"""

import numpy as np
from numba import njit

from nengine.board import OFF, SQ120
from nengine.search import PST, PST_KING_END, PST_KING_MID, evaluate


@njit(cache=False)
def score_codes(codes: np.ndarray, turn: int) -> int:
    """Return the precise ``nengine.evaluate`` score for one raw-board row."""
    board = np.full(120, OFF, dtype=np.int8)
    for square in range(64):
        board[SQ120[square]] = codes[square]
    side = 0 if turn == 1 else 1
    return evaluate(board, side, PST, PST_KING_MID, PST_KING_END)


@njit(cache=False)
def scores_batch(boards: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """Evaluate a data batch in mover-relative centipawns without approximation."""
    out = np.empty(boards.shape[0], dtype=np.int32)
    board = np.full(120, OFF, dtype=np.int8)
    for row in range(boards.shape[0]):
        for square in range(64):
            board[SQ120[square]] = boards[row, square]
        side = 0 if turns[row] == 1 else 1
        out[row] = evaluate(board, side, PST, PST_KING_MID, PST_KING_END)
    return out
