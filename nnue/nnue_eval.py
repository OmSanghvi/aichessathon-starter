"""Reference implementation of the agent-side NNUE evaluation.

Developed here so it can be tested against the training path before it is folded
into agent.py. The integer arithmetic mirrors nnue/export_weights.py exactly; see
that file for the derivation of the scales.

The hot function is numba-jitted and takes the sunfish board string directly, so
there is no python-chess object churn per node. It does a FULL accumulator
refresh (~32 row adds) rather than an incremental update, because sunfish's
Position is immutable and offers no make/unmake hook to hang a delta on. The
benchmark said a refresh costs ~4.5us against a ~30us node, which is the whole
reason this is affordable.
"""

import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ACT_MAX = 127
SHIFT = 6
W_SCALE = 64.0
CP_SCALE = 400.0

# Sunfish piece characters in python-chess piece-type order.
PIECE_ORDER = "PNBRQK"

# Sunfish's 120-char board maps to squares as: file = (i - A1) % 10,
# rank = -((i - A1) // 10), with A1 = 91. We precompute index -> square once and
# hand numba a plain int array, which is far cheaper than recomputing per node.
A1 = 91


def build_square_table() -> np.ndarray:
    table = np.full(120, -1, dtype=np.int32)
    for i in range(120):
        f = (i - A1) % 10
        r = -((i - A1) // 10)
        if 0 <= f <= 7 and 0 <= r <= 7:
            table[i] = r * 8 + f
    return table


def build_piece_table() -> np.ndarray:
    """char code -> (relative_colour * 6 + piece_type), or -1 for empty/off-board."""
    table = np.full(128, -1, dtype=np.int32)
    for t, ch in enumerate(PIECE_ORDER):
        table[ord(ch)] = t  # uppercase: the side to move
        table[ord(ch.lower())] = 6 + t  # lowercase: the opponent
    return table


@njit(cache=False)
def _forward(board_codes, square_table, piece_table, ft_w, ft_b, l1_w, l1_b, l2_w, l2_b):
    """Full-refresh integer forward pass. Returns centipawns * W_SCALE * ACT_MAX."""
    acc_n = ft_b.shape[0]
    acc = np.empty(acc_n, dtype=np.int32)
    for i in range(acc_n):
        acc[i] = ft_b[i]

    for i in range(board_codes.shape[0]):
        kind = piece_table[board_codes[i]]
        if kind < 0:
            continue
        square = square_table[i]
        if square < 0:
            continue
        feature = kind * 64 + square
        for j in range(acc_n):
            acc[j] += ft_w[feature, j]

    # Clipped ReLU. acc is already in ACT_MAX units, so saturate, do not divide.
    for i in range(acc_n):
        v = acc[i]
        if v < 0:
            acc[i] = 0
        elif v > ACT_MAX:
            acc[i] = ACT_MAX

    hidden = l1_w.shape[1]
    total = 0
    for j in range(hidden):
        s = l1_b[j]
        for i in range(acc_n):
            s += acc[i] * l1_w[i, j]
        s >>= SHIFT
        if s < 0:
            s = 0
        elif s > ACT_MAX:
            s = ACT_MAX
        total += s * l2_w[j, 0]
    total += l2_b[0]
    return total


class NnueEvaluator:
    def __init__(self, weights_path: str | Path) -> None:
        with np.load(str(weights_path)) as z:
            self.ft_w = np.ascontiguousarray(z["ft_w"], dtype=np.int32)
            self.ft_b = np.ascontiguousarray(z["ft_b"], dtype=np.int32)
            self.l1_w = np.ascontiguousarray(z["l1_w"], dtype=np.int32)
            self.l1_b = np.ascontiguousarray(z["l1_b"], dtype=np.int32)
            self.l2_w = np.ascontiguousarray(z["l2_w"], dtype=np.int32)
            self.l2_b = np.ascontiguousarray(z["l2_b"], dtype=np.int32)
        self.square_table = build_square_table()
        self.piece_table = build_piece_table()
        self._codes = np.empty(120, dtype=np.int32)

    def evaluate_board_string(self, board: str) -> int:
        # One C-level conversion, not a 120-step Python loop. The naive
        # `for i, ch in enumerate(board): codes[i] = ord(ch)` version cost about
        # as much as the whole jitted kernel and doubled the per-node price.
        codes = np.frombuffer(board.encode("latin-1"), dtype=np.uint8)
        raw = _forward(
            codes,
            self.square_table,
            self.piece_table,
            self.ft_w,
            self.ft_b,
            self.l1_w,
            self.l1_b,
            self.l2_w,
            self.l2_b,
        )
        return int(round(raw / (W_SCALE * ACT_MAX) * CP_SCALE))

    def warm(self) -> None:
        """Compile the jitted kernel at import, not on the clock."""
        blank = "." * 120
        self.evaluate_board_string(blank)
