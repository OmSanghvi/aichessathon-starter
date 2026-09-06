"""Current nengine's static evaluation, batched for residual-NNUE training.

The previous residual target used Sunfish's score.  That made the learned correction
incompatible with the mailbox engine: adding it to a different material/PST scale
lost decisively in an A/B test.  This module reproduces the evaluator that actually
ships in ``agent.py`` directly from the stored board codes, so a residual net now
has one unambiguous anchor.
"""

import numpy as np
from numba import njit

from nengine.search import (
    ENDGAME_MATERIAL,
    PIECE_VALUE,
    PST,
    PST_KING_END,
    PST_KING_MID,
)


@njit(cache=False)
def score_codes(codes: np.ndarray, turn: int) -> int:
    """Match nengine.evaluate() for one board; turn is 1 for White to move."""
    score = 0
    npm_white = 0
    npm_black = 0
    for square in range(64):
        piece = codes[square]
        if piece == 0:
            continue
        if piece <= 6:
            kind = piece
            if kind != 1 and kind != 6:
                npm_white += PIECE_VALUE[kind]
            if kind != 6:
                score += PIECE_VALUE[kind] + PST[kind, square]
        else:
            kind = piece - 6
            if kind != 1 and kind != 6:
                npm_black += PIECE_VALUE[kind]
            if kind != 6:
                score -= PIECE_VALUE[kind] + PST[kind, square ^ 56]

    endgame = npm_white + npm_black <= ENDGAME_MATERIAL
    for square in range(64):
        piece = codes[square]
        if piece == 6:
            score += PST_KING_END[square] if endgame else PST_KING_MID[square]
        elif piece == 12:
            mirror = square ^ 56
            score -= PST_KING_END[mirror] if endgame else PST_KING_MID[mirror]
    return score if turn == 1 else -score


@njit(cache=False)
def scores_batch(boards: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """Evaluate a shard in mover-relative centipawns."""
    out = np.empty(boards.shape[0], dtype=np.int32)
    for row in range(boards.shape[0]):
        out[row] = score_codes(boards[row], turns[row])
    return out
