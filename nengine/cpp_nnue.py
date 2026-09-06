"""Reference implementation of the team's C++ NNUE integer forward pass.

This is deliberately a full accumulator refresh, used to prove feature order and
integer arithmetic before the incremental Numba search integration is enabled.
The network and binary layout are from the team's pre-event C++ engine.
"""

from pathlib import Path

import numpy as np
from numba import njit

from nengine.board import EMPTY, OFF, mv_flags, mv_from, mv_to

FT_IN = 768
HL = 512
QA = 255
QB = 64
SCALE = 400
NETWORK_BYTES = FT_IN * HL * 2 + HL * 2 + 2 * HL * 2 + 4


def load_network(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int32]:
    """Load the little-endian ``net.bin`` format emitted by ``train/quant.py``."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.nbytes != NETWORK_BYTES:
        raise ValueError(f"{path} has {raw.nbytes} bytes, expected {NETWORK_BYTES}")
    offset = 0
    ft_count = FT_IN * HL
    ft = np.frombuffer(raw, "<i2", ft_count, offset).copy().reshape(FT_IN, HL)
    offset += ft_count * 2
    bias = np.frombuffer(raw, "<i2", HL, offset).copy()
    offset += HL * 2
    out = np.frombuffer(raw, "<i2", 2 * HL, offset).copy()
    offset += 2 * HL * 2
    out_bias = np.frombuffer(raw, "<i4", 1, offset)[0]
    return ft, bias, out, out_bias


@njit(cache=False)
def evaluate_full(
    board: "np.ndarray", side: int, ft: "np.ndarray", bias: "np.ndarray",
    out: "np.ndarray", out_bias: int,
) -> int:
    """C++-identical NNUE evaluation, refreshing both accumulators from board."""
    acc = np.empty((2, HL), dtype=np.int32)
    for perspective in range(2):
        for hidden in range(HL):
            acc[perspective, hidden] = bias[hidden]

    for square in range(64):
        mailbox_square = 91 + (square & 7) - 10 * (square >> 3)
        piece = board[mailbox_square]
        if piece == EMPTY or piece == OFF:  # noqa: SIM109 (Numba scalar path)
            continue
        color = 0 if piece <= 6 else 1
        piece_type = piece - 1 if piece <= 6 else piece - 7
        for perspective in range(2):
            relative_color = 0 if color == perspective else 1
            relative_square = square if perspective == 0 else square ^ 56
            feature = relative_color * 384 + piece_type * 64 + relative_square
            for hidden in range(HL):
                acc[perspective, hidden] += ft[feature, hidden]

    total = np.int64(out_bias)
    for hidden in range(HL):
        us = acc[side, hidden]
        them = acc[1 - side, hidden]
        if us < 0:
            us = 0
        elif us > QA:
            us = QA
        if them < 0:
            them = 0
        elif them > QA:
            them = QA
        total += np.int64(us) * out[hidden]
        total += np.int64(them) * out[HL + hidden]

    scaled = total * SCALE
    divisor = QA * QB
    # C++ integer division truncates toward zero; Python/Numba ``//`` floors.
    if scaled < 0:
        return int(-((-scaled) // divisor))
    return int(scaled // divisor)


@njit(cache=False)
def refresh_accumulator(
    board: "np.ndarray", ft: "np.ndarray", bias: "np.ndarray",
) -> "np.ndarray":
    """Build the two C++-layout accumulators from a mailbox board."""
    acc = np.empty((2, HL), dtype=np.int32)
    for perspective in range(2):
        for hidden in range(HL):
            acc[perspective, hidden] = bias[hidden]
    for square in range(64):
        mailbox_square = 91 + (square & 7) - 10 * (square >> 3)
        piece = board[mailbox_square]
        if piece == EMPTY or piece == OFF:  # noqa: SIM109 (Numba scalar path)
            continue
        color = 0 if piece <= 6 else 1
        piece_type = piece - 1 if piece <= 6 else piece - 7
        for perspective in range(2):
            relative_color = 0 if color == perspective else 1
            relative_square = square if perspective == 0 else square ^ 56
            feature = relative_color * 384 + piece_type * 64 + relative_square
            for hidden in range(HL):
                acc[perspective, hidden] += ft[feature, hidden]
    return acc


@njit(cache=False, inline="always")
def _accumulate(
    acc: "np.ndarray", piece: int, square120: int, ft: "np.ndarray", sign: int,
) -> None:
    square = ((98 - square120) // 10) * 8 + ((square120 - 91) % 10)
    color = 0 if piece <= 6 else 1
    piece_type = piece - 1 if piece <= 6 else piece - 7
    for perspective in range(2):
        relative_color = 0 if color == perspective else 1
        relative_square = square if perspective == 0 else square ^ 56
        feature = relative_color * 384 + piece_type * 64 + relative_square
        for hidden in range(HL):
            acc[perspective, hidden] += sign * ft[feature, hidden]


@njit(cache=False)
def apply_move(
    acc: "np.ndarray", mover: int, placed: int, captured: int, side: int, m: int,
    ft: "np.ndarray",
) -> None:
    """Apply the C++ accumulator delta after ``make_move``; reverse with -1."""
    frm, to, flags = mv_from(m), mv_to(m), mv_flags(m)
    _accumulate(acc, mover, frm, ft, -1)
    _accumulate(acc, placed, to, ft, 1)
    if captured != EMPTY and captured != OFF:
        capture_square = to + (10 if side == 0 else -10) if flags == 1 else to
        _accumulate(acc, captured, capture_square, ft, -1)
    if flags == 2:
        if to == 97:
            _accumulate(acc, 4, 98, ft, -1)
            _accumulate(acc, 4, 96, ft, 1)
        elif to == 93:
            _accumulate(acc, 4, 91, ft, -1)
            _accumulate(acc, 4, 94, ft, 1)
        elif to == 27:
            _accumulate(acc, 10, 28, ft, -1)
            _accumulate(acc, 10, 26, ft, 1)
        else:
            _accumulate(acc, 10, 21, ft, -1)
            _accumulate(acc, 10, 24, ft, 1)


@njit(cache=False)
def undo_move(
    acc: "np.ndarray", mover: int, placed: int, captured: int, side: int, m: int,
    ft: "np.ndarray",
) -> None:
    """Reverse ``apply_move`` before the mailbox move is unmade."""
    frm, to, flags = mv_from(m), mv_to(m), mv_flags(m)
    _accumulate(acc, placed, to, ft, -1)
    _accumulate(acc, mover, frm, ft, 1)
    if captured != EMPTY and captured != OFF:
        capture_square = to + (10 if side == 0 else -10) if flags == 1 else to
        _accumulate(acc, captured, capture_square, ft, 1)
    if flags == 2:
        if to == 97:
            _accumulate(acc, 4, 96, ft, -1)
            _accumulate(acc, 4, 98, ft, 1)
        elif to == 93:
            _accumulate(acc, 4, 94, ft, -1)
            _accumulate(acc, 4, 91, ft, 1)
        elif to == 27:
            _accumulate(acc, 10, 26, ft, -1)
            _accumulate(acc, 10, 28, ft, 1)
        else:
            _accumulate(acc, 10, 24, ft, -1)
            _accumulate(acc, 10, 21, ft, 1)


@njit(cache=False)
def evaluate_accumulator(
    acc: "np.ndarray", side: int, out: "np.ndarray", out_bias: int,
) -> int:
    total = np.int64(out_bias)
    for hidden in range(HL):
        us = acc[side, hidden]
        them = acc[1 - side, hidden]
        if us < 0:
            us = 0
        elif us > QA:
            us = QA
        if them < 0:
            them = 0
        elif them > QA:
            them = QA
        total += np.int64(us) * out[hidden]
        total += np.int64(them) * out[HL + hidden]
    scaled = total * SCALE
    divisor = QA * QB
    return int(-((-scaled) // divisor)) if scaled < 0 else int(scaled // divisor)
