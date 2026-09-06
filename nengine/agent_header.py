"""AI Chessathon submission: a numba-jitted negamax engine.

GENERATED FILE - edit nengine/ and run nengine/build_agent.py to regenerate.

The platform imports this module once per game and calls get_move(fen, time_left_ms).

The engine is a 10x12 mailbox move generator plus an alpha-beta search, both
numba-jitted. Its pure-Python predecessor ran at about 55,000 nodes a second and was
losing games to depth: in one lost game it needed depth 7 to see the move that saved
a won position and only reached depth 5-6 in the time it had. This reaches a few
million nodes a second, which buys several plies. IDEAS.md is explicit that numba is
how Python gets fast here and that "the gain is the depth the speed lets you afford".

Correctness is not assumed. The generator is verified by perft against python-chess
on the standard suite including Kiwipete and the en-passant and promotion traps, and
the search finds every tactic in the local suite. python-chess is used only to parse
the FEN and to validate the move on the way out, never inside the search.

Everything runs inside the platform's one core, 2 GB, no network sandbox. The jitted
functions are warmed at import so compilation lands in the 60 second init budget
rather than on the game clock.
"""

import contextlib
import time

import chess
import numpy as np
from numba import njit

EMPTY = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12
OFF = 13

N, E, S, W = -10, 1, 10, -1
CR_WK, CR_WQ, CR_BK, CR_BQ = 1, 2, 4, 8

SQ120 = np.array([91 + (s & 7) - 10 * (s >> 3) for s in range(64)], dtype=np.int16)
SQ64 = np.full(120, -1, dtype=np.int16)
for _s in range(64):
    SQ64[SQ120[_s]] = _s

OFFSETS = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-21, -19, -12, -8, 8, 12, 19, 21],
        [-11, -9, 9, 11, 0, 0, 0, 0],
        [-10, -1, 1, 10, 0, 0, 0, 0],
        [-11, -10, -9, -1, 1, 9, 10, 11],
        [-11, -10, -9, -1, 1, 9, 10, 11],
    ],
    dtype=np.int64,
)
N_OFFSETS = np.array([0, 8, 4, 4, 8, 8], dtype=np.int64)
IS_SLIDER = np.array([False, False, True, True, True, False], dtype=np.bool_)
