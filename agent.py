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
from pathlib import Path

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

# ==========================================================================
# Board representation and move generation (perft-verified against
# python-chess; see nengine/test_perft.py)
# ==========================================================================

def initial_board() -> np.ndarray:
    """Mailbox for the standard starting position."""
    b = np.full(120, OFF, dtype=np.int8)
    for s in range(64):
        b[SQ120[s]] = EMPTY
    back = [WR, WN, WB, WQ, WK, WB, WN, WR]
    for f in range(8):
        b[SQ120[f]] = back[f]
        b[SQ120[8 + f]] = WP
        b[SQ120[48 + f]] = BP
        b[SQ120[56 + f]] = back[f] + 6
    return b


# ---------------------------------------------------------------------------
# Jitted primitives
# ---------------------------------------------------------------------------


@njit(cache=False, inline="always")
def _is_white(p: int) -> bool:
    return 1 <= p <= 6


@njit(cache=False, inline="always")
def _is_black(p: int) -> bool:
    return 7 <= p <= 12


@njit(cache=False, inline="always")
def _kind(p: int) -> int:
    """Piece type 1..6 regardless of colour."""
    return p if p <= 6 else p - 6


@njit(cache=False, inline="always")
def _own(p: int, side: int) -> bool:
    return _is_white(p) if side == 0 else _is_black(p)


@njit(cache=False, inline="always")
def _enemy(p: int, side: int) -> bool:
    return _is_black(p) if side == 0 else _is_white(p)


@njit(cache=False, inline="always")
def encode(frm: int, to: int, promo: int, flags: int) -> int:
    return frm | (to << 8) | (promo << 16) | (flags << 20)


@njit(cache=False, inline="always")
def mv_from(m: int) -> int:
    return m & 0xFF


@njit(cache=False, inline="always")
def mv_to(m: int) -> int:
    return (m >> 8) & 0xFF


@njit(cache=False, inline="always")
def mv_promo(m: int) -> int:
    return (m >> 16) & 0xF


@njit(cache=False, inline="always")
def mv_flags(m: int) -> int:
    return (m >> 20) & 0xF


@njit(cache=False)
def attacked(
    board: "np.ndarray",
    sq: int,
    by_side: int,
    offsets: "np.ndarray",
    n_offsets: "np.ndarray",
    is_slider: "np.ndarray",
) -> bool:
    """Is mailbox square `sq` attacked by `by_side`?

    Used for legality (is our king attacked after the move) and for check
    detection, so it runs constantly and stays deliberately simple.
    """
    # Pawns. White pawns attack "north" (-10) so they sit south-east/south-west
    # of the square they attack.
    if by_side == 0:
        if board[sq + 9] == WP or board[sq + 11] == WP:
            return True
    else:
        if board[sq - 9] == BP or board[sq - 11] == BP:
            return True

    # Knights, king: single steps.
    for kind in (2, 6):
        want = kind if by_side == 0 else kind + 6
        n = n_offsets[kind - 1]
        for k in range(n):
            t = sq + offsets[kind - 1, k]
            if board[t] == want:
                return True

    # Sliders: bishop/queen on diagonals, rook/queen on files and ranks.
    for kind in (3, 4):
        n = n_offsets[kind - 1]
        for k in range(n):
            d = offsets[kind - 1, k]
            t = sq + d
            while True:
                p = board[t]
                if p == OFF:
                    break
                if p != EMPTY:
                    if by_side == 0:
                        if p == kind or p == WQ:  # noqa: SIM109 (tuple `in` slower in numba)
                            return True
                    elif p == kind + 6 or p == BQ:
                        return True
                    break
                t += d
    return False


@njit(cache=False)
def gen_moves(
    board: "np.ndarray",
    side: int,
    castling: int,
    ep: int,
    out: "np.ndarray",
    offsets: "np.ndarray",
    n_offsets: "np.ndarray",
    is_slider: "np.ndarray",
) -> int:
    """Generate pseudo-legal moves into `out`; return how many."""
    n = 0
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if p == EMPTY or p == OFF:  # noqa: SIM109 (explicit compares; tuple `in` is slower under numba)
            continue
        if not _own(p, side):
            continue
        kind = _kind(p)

        if kind == 1:  # pawn
            fwd = N if side == 0 else S
            start_rank_lo = 81 if side == 0 else 31
            start_rank_hi = 88 if side == 0 else 38
            promo_lo = 21 if side == 0 else 91
            promo_hi = 28 if side == 0 else 98

            t = sq + fwd
            if board[t] == EMPTY:
                if promo_lo <= t <= promo_hi:
                    base = 0 if side == 0 else 6
                    for pc in (WN, WB, WR, WQ):
                        out[n] = encode(sq, t, pc + base, 0)
                        n += 1
                else:
                    out[n] = encode(sq, t, 0, 0)
                    n += 1
                    if start_rank_lo <= sq <= start_rank_hi:
                        t2 = t + fwd
                        if board[t2] == EMPTY:
                            out[n] = encode(sq, t2, 0, 0)
                            n += 1
            for dc in (fwd - 1, fwd + 1):
                t = sq + dc
                q = board[t]
                if q == OFF:
                    continue
                if _enemy(q, side):
                    if promo_lo <= t <= promo_hi:
                        base = 0 if side == 0 else 6
                        for pc in (WN, WB, WR, WQ):
                            out[n] = encode(sq, t, pc + base, 0)
                            n += 1
                    else:
                        out[n] = encode(sq, t, 0, 0)
                        n += 1
                elif ep != 0 and t == ep:
                    out[n] = encode(sq, t, 0, 1)
                    n += 1
            continue

        # Everything else: offsets, sliding where applicable.
        cnt = n_offsets[kind - 1]
        slide = is_slider[kind - 1]
        for k in range(cnt):
            d = offsets[kind - 1, k]
            t = sq + d
            while True:
                q = board[t]
                if q == OFF:
                    break
                if q == EMPTY:
                    out[n] = encode(sq, t, 0, 0)
                    n += 1
                    if not slide:
                        break
                    t += d
                    continue
                if _enemy(q, side):
                    out[n] = encode(sq, t, 0, 0)
                    n += 1
                break

    # Castling. The squares between must be empty and the king may not start on,
    # cross, or land on an attacked square. Written as flat conditions so the
    # short-circuit order puts the cheap emptiness tests before the expensive
    # attack scans.
    if side == 0:
        if (
            (castling & CR_WK) != 0
            and board[96] == EMPTY
            and board[97] == EMPTY
            and not attacked(board, 95, 1, offsets, n_offsets, is_slider)
            and not attacked(board, 96, 1, offsets, n_offsets, is_slider)
            and not attacked(board, 97, 1, offsets, n_offsets, is_slider)
        ):
            out[n] = encode(95, 97, 0, 2)
            n += 1
        if (
            (castling & CR_WQ) != 0
            and board[94] == EMPTY
            and board[93] == EMPTY
            and board[92] == EMPTY
            and not attacked(board, 95, 1, offsets, n_offsets, is_slider)
            and not attacked(board, 94, 1, offsets, n_offsets, is_slider)
            and not attacked(board, 93, 1, offsets, n_offsets, is_slider)
        ):
            out[n] = encode(95, 93, 0, 2)
            n += 1
    else:
        if (
            (castling & CR_BK) != 0
            and board[26] == EMPTY
            and board[27] == EMPTY
            and not attacked(board, 25, 0, offsets, n_offsets, is_slider)
            and not attacked(board, 26, 0, offsets, n_offsets, is_slider)
            and not attacked(board, 27, 0, offsets, n_offsets, is_slider)
        ):
            out[n] = encode(25, 27, 0, 2)
            n += 1
        if (
            (castling & CR_BQ) != 0
            and board[24] == EMPTY
            and board[23] == EMPTY
            and board[22] == EMPTY
            and not attacked(board, 25, 0, offsets, n_offsets, is_slider)
            and not attacked(board, 24, 0, offsets, n_offsets, is_slider)
            and not attacked(board, 23, 0, offsets, n_offsets, is_slider)
        ):
            out[n] = encode(25, 23, 0, 2)
            n += 1
    return n


@njit(cache=False)
def make_move(
    board: "np.ndarray",
    side: int,
    castling: int,
    ep: int,
    m: int,
) -> "tuple[int, int, int]":
    """Apply `m`. Returns (captured, new_castling, new_ep) for unmaking."""
    frm = mv_from(m)
    to = mv_to(m)
    promo = mv_promo(m)
    flags = mv_flags(m)
    p = board[frm]
    captured = board[to]

    board[to] = p if promo == 0 else promo
    board[frm] = EMPTY

    if flags == 1:  # en passant: the captured pawn is not on `to`
        victim_sq = to + (S if side == 0 else N)
        captured = board[victim_sq]
        board[victim_sq] = EMPTY
    elif flags == 2:  # castling: move the rook too
        if to == 97:
            board[98] = EMPTY
            board[96] = WR
        elif to == 93:
            board[91] = EMPTY
            board[94] = WR
        elif to == 27:
            board[28] = EMPTY
            board[26] = BR
        else:
            board[21] = EMPTY
            board[24] = BR

    new_cr = castling
    if p == WK:
        new_cr &= ~(CR_WK | CR_WQ)
    elif p == BK:
        new_cr &= ~(CR_BK | CR_BQ)
    if frm == 98 or to == 98:
        new_cr &= ~CR_WK
    if frm == 91 or to == 91:
        new_cr &= ~CR_WQ
    if frm == 28 or to == 28:
        new_cr &= ~CR_BK
    if frm == 21 or to == 21:
        new_cr &= ~CR_BQ

    new_ep = 0
    if _kind(p) == 1 and abs(to - frm) == 20:
        new_ep = (frm + to) // 2

    return captured, new_cr, new_ep


@njit(cache=False)
def unmake_move(board: "np.ndarray", side: int, m: int, captured: int) -> None:
    """Undo `m`, restoring `captured`."""
    frm = mv_from(m)
    to = mv_to(m)
    promo = mv_promo(m)
    flags = mv_flags(m)

    p = board[to]
    if promo != 0:
        p = WP if side == 0 else BP
    board[frm] = p

    if flags == 1:
        board[to] = EMPTY
        victim_sq = to + (S if side == 0 else N)
        board[victim_sq] = captured
    else:
        board[to] = captured
        if flags == 2:
            if to == 97:
                board[98] = WR
                board[96] = EMPTY
            elif to == 93:
                board[91] = WR
                board[94] = EMPTY
            elif to == 27:
                board[28] = BR
                board[26] = EMPTY
            else:
                board[21] = BR
                board[24] = EMPTY


@njit(cache=False)
def find_king(board: "np.ndarray", side: int) -> int:
    want = WK if side == 0 else BK
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        if board[sq] == want:
            return sq
    return -1


@njit(cache=False)
def in_check(
    board: "np.ndarray",
    side: int,
    offsets: "np.ndarray",
    n_offsets: "np.ndarray",
    is_slider: "np.ndarray",
) -> bool:
    k = find_king(board, side)
    if k < 0:
        return False
    return attacked(board, k, 1 - side, offsets, n_offsets, is_slider)


@njit(cache=False)
def perft(
    board: "np.ndarray",
    side: int,
    castling: int,
    ep: int,
    depth: int,
    offsets: "np.ndarray",
    n_offsets: "np.ndarray",
    is_slider: "np.ndarray",
) -> int:
    """Count legal leaf nodes. The correctness gate for the whole engine."""
    if depth == 0:
        return 1
    buf = np.empty(256, dtype=np.int32)
    n = gen_moves(board, side, castling, ep, buf, offsets, n_offsets, is_slider)
    total = 0
    for idx in range(n):
        m = buf[idx]
        captured, new_cr, new_ep = make_move(board, side, castling, ep, m)
        if not in_check(board, side, offsets, n_offsets, is_slider):
            total += perft(board, 1 - side, new_cr, new_ep, depth - 1,
                           offsets, n_offsets, is_slider)
        unmake_move(board, side, m, captured)
    return total

# ==========================================================================
# Candidate NNUE evaluator ported from the team's pre-event C++ source
# ==========================================================================

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

# ==========================================================================
# Search: alpha-beta, transposition table, quiescence, move ordering
# ==========================================================================

MATE = 30000
INF = 32000
MAX_PLY = 64


# Material, in centipawns, indexed by piece type 1..6.
PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int32)

# Split movement directions by slider class for the mailbox SEE. These are
# constants rather than slices of ``OFFSETS`` so Numba keeps the inner scan
# simple and monomorphic.
N_OFFSETS_SEARCH = np.array([-21, -19, -12, -8, 8, 12, 19, 21], dtype=np.int8)
BISHOP_OFFSETS = np.array([-11, -9, 9, 11], dtype=np.int8)
ROOK_OFFSETS = np.array([-10, -1, 1, 10], dtype=np.int8)
KING_OFFSETS_SEARCH = np.array([-11, -10, -9, -1, 1, 9, 10, 11], dtype=np.int8)

# Piece-square tables from White's point of view, a1..h8, mirrored for Black.
_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_QUEEN = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING_MID = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
_KING_END = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]

# PST[kind, square64] for White; Black reads square64 ^ 56.
PST = np.zeros((7, 64), dtype=np.int32)
for _k, _t in ((1, _PAWN), (2, _KNIGHT), (3, _BISHOP), (4, _ROOK), (5, _QUEEN)):
    PST[_k] = np.array(_t, dtype=np.int32)
PST_KING_MID = np.array(_KING_MID, dtype=np.int32)
PST_KING_END = np.array(_KING_END, dtype=np.int32)

# Tapered PeSTO evaluation from this team's pre-event C++ engine. The source
# tables are a8-first; mailbox square indices are a1-first, so White uses
# ``square ^ 56`` and Black uses ``square``. A material phase blends middlegame
# and endgame scores rather than abruptly switching the whole evaluation.
MG_VALUE = np.array([0, 82, 337, 365, 477, 1025, 0], dtype=np.int32)
EG_VALUE = np.array([0, 94, 281, 297, 512, 936, 0], dtype=np.int32)
PHASE_VALUE = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
MG_PESTO = np.array([
0,0,0,0,0,0,0,0,98,134,61,95,68,126,34,-11,-6,7,26,31,65,56,25,-20,-14,13,6,21,23,12,17,-23,-27,-2,-5,12,17,6,10,-25,-26,-4,-4,-10,3,3,33,-12,-35,-1,-20,-23,-15,24,38,-22,0,0,0,0,0,0,0,0,
-167,-89,-34,-49,61,-97,-15,-107,-73,-41,72,36,23,62,7,-17,-47,60,37,65,84,129,73,44,-9,17,19,53,37,69,18,22,-13,4,16,13,28,19,21,-8,-23,-9,12,10,19,17,25,-16,-29,-53,-12,-3,-1,18,-14,-19,-105,-21,-58,-33,-17,-28,-19,-23,
-29,4,-82,-37,-25,-42,7,-8,-26,16,-18,-13,30,59,18,-47,-16,37,43,40,35,50,37,-2,-4,5,19,50,37,37,7,-2,-6,13,13,26,34,12,10,4,0,15,15,15,14,27,18,10,4,15,16,0,7,21,33,1,-33,-3,-14,-21,-13,-12,-39,-21,
32,42,32,51,63,9,31,43,27,32,58,62,80,67,26,44,-5,19,26,36,17,45,61,16,-24,-11,7,26,24,35,-8,-20,-36,-26,-12,-1,9,-7,6,-23,-45,-25,-16,-17,3,0,-5,-33,-44,-16,-20,-9,-1,11,-6,-71,-19,-13,1,17,16,7,-37,-26,
-28,0,29,12,59,44,43,45,-24,-39,-5,1,-16,57,28,54,-13,-17,7,8,29,56,47,57,-27,-27,-16,-16,-1,17,-2,1,-9,-26,-9,-10,-2,-4,3,-3,-14,2,-11,-2,-5,2,14,5,-35,-8,11,2,8,15,-3,1,-1,-18,-9,10,-15,-25,-31,-50,
-65,23,16,-15,-56,-34,2,13,29,-1,-20,-7,-8,-4,-38,-29,-9,24,2,-16,-20,6,22,-22,-17,-20,-12,-27,-30,-25,-14,-36,-49,-1,-27,-39,-46,-44,-33,-51,-14,-14,-22,-46,-44,-30,-15,-27,1,7,-8,-64,-43,-16,9,8,-15,36,12,-54,8,-28,24,14,
], dtype=np.int32).reshape(6, 64)
EG_PESTO = np.array([
0,0,0,0,0,0,0,0,178,173,158,134,147,132,165,187,94,100,85,67,56,53,82,84,32,24,13,5,-2,4,17,17,13,9,-3,-7,-7,-8,3,-1,4,7,-6,1,0,-5,-1,-8,13,8,8,10,13,0,2,-7,0,0,0,0,0,0,0,0,
-58,-38,-13,-28,-31,-27,-63,-99,-25,-8,-25,-2,-9,-25,-24,-52,-24,-20,10,9,-1,-9,-19,-41,-17,3,22,22,22,11,8,-18,-18,-6,16,25,16,17,4,-18,-23,-3,-1,15,10,-3,-20,-22,-42,-20,-10,-5,-2,-20,-23,-44,-29,-51,-23,-15,-22,-18,-50,-64,
-14,-21,-11,-8,-7,-9,-17,-24,-8,-4,7,-12,-3,-13,-4,-14,2,-8,0,-1,-2,6,0,4,-3,9,12,9,14,10,3,2,-6,3,13,19,7,10,-3,-9,-12,-3,8,10,13,3,-7,-15,-14,-18,-7,-1,4,-9,-15,-27,-23,-9,-23,-5,-9,-16,-5,-17,
13,10,18,15,12,12,8,5,11,13,13,11,-3,3,8,3,7,7,7,5,4,-3,-5,-3,4,3,13,1,2,1,-1,2,3,5,8,4,-5,-6,-8,-11,-4,0,-5,-1,-7,-12,-8,-16,-6,-6,0,2,-9,-9,-11,-3,-9,2,3,-1,-5,-13,4,-20,
-9,22,22,27,27,19,10,20,-17,20,32,41,58,25,30,0,-20,6,9,49,47,35,19,9,3,22,24,45,57,40,57,36,-18,28,19,47,31,34,39,23,-16,-27,15,6,9,17,10,5,-22,-23,-30,-16,-16,-23,-36,-32,-33,-28,-22,-43,-5,-32,-20,-41,
-74,-35,-18,-18,-11,15,4,-17,-12,17,14,17,17,38,23,11,10,17,23,15,20,45,44,13,-8,22,24,27,26,33,26,3,-18,-4,21,24,27,23,9,-11,-19,-3,11,21,23,16,7,-9,-27,-11,4,13,14,4,-5,-17,-53,-34,-21,-11,-28,-14,-24,-43,
], dtype=np.int32).reshape(6, 64)

# Non-pawn material below which the king centralises.
ENDGAME_MATERIAL = 1300

# Small, deliberately conservative positional terms. They are integer-only scans
# of the mailbox, so they cost far less than an extra searched ply. Keeping them
# below a pawn prevents static opinion from overriding tactics.
PASSED_PAWN_BONUS = np.array([0, 0, 4, 10, 22, 40, 70, 0], dtype=np.int32)
PROTECTED_PASSER_BONUS = 8
# Knight and bishop freedom is a stable positional signal. Rook and queen reach
# is more tactical, so search evaluates it instead of ray-counting at every leaf.
MOBILITY_BONUS = np.array([0, 0, 2, 1, 0, 0, 0], dtype=np.int32)
MISSING_SHIELD_PENALTY = 11
SECOND_SHIELD_PENALTY = 3
OPEN_KING_FILE_PENALTY = 7
# A pair of bishops retains pressure across both colour complexes.  PeSTO's
# individual piece-square entries do not express the *pair* interaction, which
# made the engine too willing to release a blocked enemy knight by exchanging a
# bishop.  Keep this modest: it is a positional tie-breaker, not a substitute for
# tactical search.
BISHOP_PAIR_BONUS = 35

# For each colour and square, squares on which an enemy pawn would stop a
# passer. This makes passed-pawn testing a uint64 mask test at evaluation time.
PASSED_PAWN_MASK = np.zeros((2, 64), dtype=np.uint64)
for _side in range(2):
    for _square in range(64):
        _rank, _file = divmod(_square, 8)
        _mask = np.uint64(0)
        _rank_range = range(_rank + 1, 8) if _side == 0 else range(_rank - 1, -1, -1)
        for _target_rank in _rank_range:
            for _target_file in range(max(0, _file - 1), min(7, _file + 1) + 1):
                _mask |= np.uint64(1) << np.uint64(_target_rank * 8 + _target_file)
        PASSED_PAWN_MASK[_side, _square] = _mask


# Zobrist keys. Fixed seed so a run is reproducible and A/B tests are honest.
_rng = np.random.default_rng(0xC0FFEE)
ZOB_PIECE = _rng.integers(1, 2**63 - 1, size=(13, 120), dtype=np.int64)
ZOB_SIDE = int(_rng.integers(1, 2**63 - 1, dtype=np.int64))
ZOB_CASTLE = _rng.integers(1, 2**63 - 1, size=16, dtype=np.int64)
ZOB_EP = _rng.integers(1, 2**63 - 1, size=120, dtype=np.int64)
# The ordinary position hash intentionally excludes the FEN halfmove clock for
# repetition detection.  The TT, however, must distinguish it near a 50-move
# draw, so it uses this extra key component locally at probe/store time.
ZOB_HALF = _rng.integers(1, 2**63 - 1, size=101, dtype=np.int64)

TT_BITS = 21  # 2M entries, about 40 MB across the arrays
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# A very narrow root safety rail for obvious, non-checking queen-scale hangs.
# It deliberately does not suppress normal speculative exchanges or sacrifices;
# the previous broad filter lost badly against the unfiltered engine.  Checks,
# promotions, and check evasions are always left to the main search.
ROOT_CATASTROPHIC_SEE = 500


@njit(cache=False, inline="always")
def _score_to_tt(score: int, ply: int) -> int:
    """Convert a mate score to a node-relative score for table storage.

    Mate scores carry the distance from the root (``MATE - ply``).  A position
    may be reached at a different ply through a transposition or on a later
    move, so storing that root-relative number directly makes a shorter mate
    look longer, or vice versa, when it is probed again.
    """
    if score >= MATE - MAX_PLY:
        return score + ply
    if score <= -MATE + MAX_PLY:
        return score - ply
    return score


@njit(cache=False, inline="always")
def _score_from_tt(score: int, ply: int) -> int:
    """Restore a node-relative mate score after a transposition-table probe."""
    if score >= MATE - MAX_PLY:
        return score - ply
    if score <= -MATE + MAX_PLY:
        return score + ply
    return score


def new_tt() -> tuple[
    "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"
]:
    """Open-addressed transposition table as parallel numpy arrays."""
    return (
        np.zeros(TT_SIZE, dtype=np.int64),  # key
        np.zeros(TT_SIZE, dtype=np.int32),  # score
        np.zeros(TT_SIZE, dtype=np.int32),  # move
        np.zeros(TT_SIZE, dtype=np.int8),  # depth
        np.zeros(TT_SIZE, dtype=np.int8),  # flag
    )


@njit(cache=False)
def zobrist(
    board: "np.ndarray",
    side: int,
    castling: int,
    ep: int,
    zp: "np.ndarray",
    zs: int,
    zc: "np.ndarray",
    ze: "np.ndarray",
) -> int:
    h = np.int64(0)
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if p != EMPTY and p != OFF:
            h ^= zp[p, sq]
    if side == 1:
        h ^= zs
    h ^= zc[castling & 15]
    if ep != 0:
        h ^= ze[ep]
    return int(h)


@njit(cache=False, inline="always")
def zobrist_after_move(
    h: int, board_after: "np.ndarray", side: int, castling: int, ep: int,
    m: int, captured: int, new_castling: int, new_ep: int,
    zp: "np.ndarray", zs: int, zc: "np.ndarray", ze: "np.ndarray",
) -> int:
    """Update a Zobrist key after ``make_move`` without scanning the board."""
    frm = mv_from(m)
    to = mv_to(m)
    flags = mv_flags(m)
    placed = board_after[to]
    mover = placed
    if mv_promo(m) != 0:
        mover = WP if side == 0 else BP

    updated = np.int64(h)
    updated ^= zp[mover, frm]
    updated ^= zp[placed, to]
    if flags == 1:
        victim_square = to + (S if side == 0 else -S)
        updated ^= zp[captured, victim_square]
    elif captured != EMPTY and captured != OFF:
        updated ^= zp[captured, to]

    if flags == 2:
        if to == 97:
            updated ^= zp[WR, 98] ^ zp[WR, 96]
        elif to == 93:
            updated ^= zp[WR, 91] ^ zp[WR, 94]
        elif to == 27:
            updated ^= zp[BR, 28] ^ zp[BR, 26]
        else:
            updated ^= zp[BR, 21] ^ zp[BR, 24]

    updated ^= zc[castling & 15] ^ zc[new_castling & 15]
    if ep != 0:
        updated ^= ze[ep]
    if new_ep != 0:
        updated ^= ze[new_ep]
    updated ^= zs
    return int(updated)


@njit(cache=False, inline="always")
def zobrist_after_null(h: int, ep: int, zs: int, ze: "np.ndarray") -> int:
    """Update a key for the search-only null move (side flip, no ep square)."""
    updated = np.int64(h) ^ zs
    if ep != 0:
        updated ^= ze[ep]
    return int(updated)


@njit(cache=False, inline="always")
def next_halfmove_clock(mover: int, captured: int, halfmove_clock: int) -> int:
    """Return the FEN halfmove clock after a searched move.

    The 50-move rule resets on every pawn move or capture, including en passant
    and promotion.  Keeping this state in the recursive search makes a drawn
    branch score as a draw instead of an invented win or loss.
    """
    if mover == 1 or mover == 7 or (captured != EMPTY and captured != OFF):
        return 0
    return halfmove_clock + 1


@njit(cache=False, inline="always")
def _mobility(board: "np.ndarray", sq: int, kind: int, side: int) -> int:
    """Count pseudo-legal destinations for a non-pawn, non-king piece."""
    count = 0
    for offset_index in range(N_OFFSETS[kind - 1]):
        direction = OFFSETS[kind - 1, offset_index]
        target = sq + direction
        while board[target] != OFF:
            occupant = board[target]
            if occupant == EMPTY:
                count += 1
                if IS_SLIDER[kind - 1]:
                    target += direction
                    continue
            elif (occupant <= 6) != (side == 0):
                count += 1
            break
    return count


@njit(cache=False, inline="always")
def _king_safety(board: "np.ndarray", king_sq: int, side: int) -> int:
    """Return pawn-shield and open-file danger near an unadvanced king."""
    rank = (98 - king_sq) // 10
    if (side == 0 and rank > 2) or (side == 1 and rank < 5):
        return 0

    own_pawn = 1 if side == 0 else 7
    forward = -10 if side == 0 else 10
    file_index = (king_sq - 91) % 10
    penalty = 0
    for file_delta in range(-1, 2):
        target_file = file_index + file_delta
        if target_file < 0 or target_file > 7:
            continue
        if board[king_sq + forward + file_delta] != own_pawn:
            penalty += MISSING_SHIELD_PENALTY
        if board[king_sq + 2 * forward + file_delta] != own_pawn:
            penalty += SECOND_SHIELD_PENALTY

        has_pawn_on_file = False
        for pawn_rank in range(8):
            pawn_sq = 91 + target_file - 10 * pawn_rank
            if board[pawn_sq] == own_pawn:
                has_pawn_on_file = True
                break
        if not has_pawn_on_file:
            penalty += OPEN_KING_FILE_PENALTY
    return penalty


@njit(cache=False)
def evaluate(
    board: "np.ndarray",
    side: int,
    pst: "np.ndarray",
    king_mid: "np.ndarray",
    king_end: "np.ndarray",
) -> int:
    """Static evaluation from the side-to-move's point of view."""
    score = 0
    mg_score = 0
    eg_score = 0
    phase = 0
    npm_w = 0
    npm_b = 0
    white_pawns = np.uint64(0)
    black_pawns = np.uint64(0)
    white_king = 0
    black_king = 0
    white_bishops = 0
    black_bishops = 0
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if p == EMPTY or p == OFF:  # noqa: SIM109 (explicit compares; tuple `in` is slower under numba)
            continue
        if p <= 6:
            kind = p
            if kind != 1 and kind != 6:
                npm_w += PIECE_VALUE[kind]
            table_square = s ^ 56
            mg_score += MG_VALUE[kind] + MG_PESTO[kind - 1, table_square]
            eg_score += EG_VALUE[kind] + EG_PESTO[kind - 1, table_square]
            phase += PHASE_VALUE[kind]
            if kind == 6:
                white_king = sq
            elif kind == 3:
                white_bishops += 1

            if kind == 1:
                white_pawns |= np.uint64(1) << np.uint64(s)
            elif kind != 6:
                mobility_weight = MOBILITY_BONUS[kind]
                if mobility_weight != 0:
                    score += mobility_weight * _mobility(board, sq, kind, 0)
        else:
            kind = p - 6
            if kind != 1 and kind != 6:
                npm_b += PIECE_VALUE[kind]
            mg_score -= MG_VALUE[kind] + MG_PESTO[kind - 1, s]
            eg_score -= EG_VALUE[kind] + EG_PESTO[kind - 1, s]
            phase += PHASE_VALUE[kind]
            if kind == 6:
                black_king = sq
            elif kind == 3:
                black_bishops += 1

            if kind == 1:
                black_pawns |= np.uint64(1) << np.uint64(s)
            elif kind != 6:
                mobility_weight = MOBILITY_BONUS[kind]
                if mobility_weight != 0:
                    score -= mobility_weight * _mobility(board, sq, kind, 1)

    if phase > 24:
        phase = 24
    score = (mg_score * phase + eg_score * (24 - phase)) // 24
    if white_bishops >= 2:
        score += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        score -= BISHOP_PAIR_BONUS
    endgame = (npm_w + npm_b) <= ENDGAME_MATERIAL
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if p == 1 and (black_pawns & PASSED_PAWN_MASK[0, s]) == 0:
            score += PASSED_PAWN_BONUS[s >> 3]
            if board[sq + 9] == 1 or board[sq + 11] == 1:
                score += PROTECTED_PASSER_BONUS
        elif p == 7 and (white_pawns & PASSED_PAWN_MASK[1, s]) == 0:
            score -= PASSED_PAWN_BONUS[7 - (s >> 3)]
            if board[sq - 9] == 7 or board[sq - 11] == 7:
                score -= PROTECTED_PASSER_BONUS

    if not endgame:
        score -= _king_safety(board, white_king, 0)
        score += _king_safety(board, black_king, 1)

    return score if side == 0 else -score


@njit(cache=False)
def _has_non_pawn_material(board: "np.ndarray", side: int) -> bool:
    """Does `side` have a knight, bishop, rook or queen? Null-move pruning is
    unsound in king-and-pawn endings (zugzwang), where passing is artificially
    good, so it is gated on this."""
    # Codes: white N B R Q = 2..5, black n b r q = 8..11.
    if side == 0:
        lo, hi = 2, 5
    else:
        lo, hi = 8, 11
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if lo <= p <= hi:
            return True
    return False


@njit(cache=False, inline="always")
def _mvv_lva(board: "np.ndarray", m: int) -> int:
    """Capture score: most valuable victim, least valuable attacker."""
    victim = board[mv_to(m)]
    if victim == EMPTY or victim == OFF:  # noqa: SIM109 (explicit compares; tuple `in` is slower under numba)
        vv = 0
    else:
        vv = PIECE_VALUE[victim if victim <= 6 else victim - 6]
    attacker = board[mv_from(m)]
    av = PIECE_VALUE[attacker if attacker <= 6 else attacker - 6]
    promo = mv_promo(m)
    pv = 0 if promo == 0 else PIECE_VALUE[promo if promo <= 6 else promo - 6]
    return int(vv * 16 - av + pv * 16)


@njit(cache=False, inline="always")
def _least_attacker(board: "np.ndarray", target: int, side: int) -> int:
    """Return the square of ``side``'s cheapest pseudo-legal attacker.

    This is the mailbox analogue of the bitboard attacker scan in the team's C++
    SEE. Looking outwards from the exchange square exposes x-ray sliders after
    every recapture without rebuilding an attack map.
    """
    pawn = 1 if side == 0 else 7
    if side == 0:
        if board[target + 9] == pawn:
            return target + 9
        if board[target + 11] == pawn:
            return target + 11
    else:
        if board[target - 9] == pawn:
            return target - 9
        if board[target - 11] == pawn:
            return target - 11

    knight = 2 if side == 0 else 8
    for k in range(8):
        square = target + N_OFFSETS_SEARCH[k]
        if board[square] == knight:
            return int(square)

    bishop = 3 if side == 0 else 9
    rook = 4 if side == 0 else 10
    queen = 5 if side == 0 else 11
    queen_square = 0
    for k in range(4):
        direction = BISHOP_OFFSETS[k]
        square = target + direction
        while board[square] == EMPTY:
            square += direction
        piece = board[square]
        if piece == bishop:
            return int(square)
        if piece == queen:
            queen_square = square
    for k in range(4):
        direction = ROOK_OFFSETS[k]
        square = target + direction
        while board[square] == EMPTY:
            square += direction
        piece = board[square]
        if piece == rook:
            return int(square)
        if piece == queen:
            queen_square = square
    if queen_square != 0:
        return queen_square

    king = 6 if side == 0 else 12
    for k in range(8):
        square = target + KING_OFFSETS_SEARCH[k]
        if board[square] == king:
            return int(square)
    return 0


@njit(cache=False)
def see(
    board: "np.ndarray", side: int, castling: int, ep: int, m: int,
    offsets: "np.ndarray", n_offsets: "np.ndarray", is_slider: "np.ndarray",
) -> int:
    """Static exchange evaluation for a capture, from the mover's perspective.

    The routine follows least-valuable recaptures on one square, revealing slider
    x-rays as the square empties, then minimaxes the material gains backwards.
    As in the original C++ version, it deliberately models material only; it is
    used for move ordering and conservative quiescence pruning, not as an eval.
    """
    target = mv_to(m)
    victim = board[target]
    if mv_flags(m) == 1:
        victim = 7 if side == 0 else 1
    if victim == EMPTY or victim == OFF:  # noqa: SIM109 (explicit compares are faster in Numba)
        return 0

    gains = np.empty(32, dtype=np.int32)
    moves = np.empty(32, dtype=np.int32)
    captured_stack = np.empty(32, dtype=np.int8)
    sides = np.empty(32, dtype=np.int8)
    victim_kind = victim if victim <= 6 else victim - 6
    gains[0] = PIECE_VALUE[victim_kind]

    captured, next_castling, next_ep = make_move(board, side, castling, ep, m)
    moves[0] = m
    captured_stack[0] = captured
    sides[0] = side
    count = 1
    depth = 0
    recapturing_side = 1 - side

    while count < 32:
        attacker_square = _least_attacker(board, target, recapturing_side)
        if attacker_square == 0:
            break
        attacker = board[attacker_square]
        attacker_kind = attacker if attacker <= 6 else attacker - 6
        previous_attacker = board[target]
        previous_kind = (
            previous_attacker if previous_attacker <= 6 else previous_attacker - 6
        )
        gains[depth + 1] = PIECE_VALUE[previous_kind] - gains[depth]
        recapture = encode(attacker_square, target, 0, 0)
        captured, next_castling, next_ep = make_move(
            board, recapturing_side, next_castling, next_ep, recapture
        )
        # A king may only enter an unattacked exchange square. Other pinned
        # attackers are intentionally treated like the C++ SEE: an ordering
        # approximation, never a source of a search cutoff by itself.
        if attacker_kind == 6 and in_check(board, recapturing_side, offsets, n_offsets, is_slider):
            unmake_move(board, recapturing_side, recapture, captured)
            break
        moves[count] = recapture
        captured_stack[count] = captured
        sides[count] = recapturing_side
        count += 1
        depth += 1
        recapturing_side = 1 - recapturing_side

    while depth > 0:
        previous = gains[depth - 1]
        if gains[depth] > -previous:
            gains[depth - 1] = -gains[depth]
        depth -= 1

    for index in range(count - 1, -1, -1):
        unmake_move(board, sides[index], moves[index], captured_stack[index])
    return int(gains[0])


@njit(cache=False, inline="always")
def _is_third_repetition(
    h: int, game_hashes: "np.ndarray", game_count: int,
    path_hashes: "np.ndarray", path_count: int,
) -> bool:
    """Would adding ``h`` make this search line a threefold repetition?"""
    occurrences = 0
    for index in range(game_count):
        if game_hashes[index] == h:
            occurrences += 1
    for index in range(path_count):
        if path_hashes[index] == h:
            occurrences += 1
    return occurrences >= 2


@njit(cache=False, inline="always")
def late_move_reduction(depth: int, move_number: int) -> int:
    """One-ply reduction for late quiet moves.

    The depth-scaled experiment is deliberately disabled until it wins a
    meaningful arena sample. Full re-search remains the safety net when this
    reduced probe unexpectedly raises alpha.
    """
    return 0


@njit(cache=False)
def _order(
    board: "np.ndarray",
    buf: "np.ndarray",
    n: int,
    scores: "np.ndarray",
    tt_move: int,
    killers: "np.ndarray",
    ply: int,
    history: "np.ndarray",
    side: int,
    castling: int,
    ep: int,
    offsets: "np.ndarray",
    n_offsets: "np.ndarray",
    is_slider: "np.ndarray",
) -> None:
    """Score each move for ordering; the caller does selection sort on the fly."""
    for i in range(n):
        m = buf[i]
        if m == tt_move and tt_move != 0:
            scores[i] = 1 << 28
            continue
        victim = board[mv_to(m)]
        if (victim != EMPTY and victim != OFF) or mv_promo(m) != 0 or mv_flags(m) == 1:
            if mv_promo(m) != 0 or see(
                board, side, castling, ep, m, offsets, n_offsets, is_slider
            ) >= 0:
                scores[i] = (1 << 24) + _mvv_lva(board, m)
            else:
                # Retain losing captures for tactical completeness, but place
                # them behind killer quiets. This is the C++ engine's SEE-aware
                # ordering policy expressed in the mailbox score bands.
                scores[i] = (1 << 20) + _mvv_lva(board, m)
            continue
        if ply < MAX_PLY and (m == killers[ply, 0] or m == killers[ply, 1]):
            scores[i] = 1 << 22
            continue
        scores[i] = history[side, mv_from(m), mv_to(m)]


@njit(cache=False)
def quiesce(
    board: "np.ndarray", side: int, castling: int, ep: int, alpha: int, beta: int,
    offsets: "np.ndarray", n_offsets: "np.ndarray", is_slider: "np.ndarray",
    pst: "np.ndarray", king_mid: "np.ndarray", king_end: "np.ndarray",
    counters: "np.ndarray", node_limit: int, halfmove_clock: int, qply: int,
    nnue_acc: "np.ndarray", nnue_ft: "np.ndarray", nnue_out: "np.ndarray",
    nnue_out_bias: int, use_nnue: bool,
) -> int:
    """Resolve forcing leaf positions without allowing an illegal stand-pat.

    A side in check has no right to use its static evaluation: it must first make a
    legal evasion.  Treating that position as quiet creates especially bad errors in
    king attacks, because the search can stop one ply before a forced defence.

    Every q-node participates in the same node budget as the main search.  Previously
    a capture sequence could exceed the driver's budget without setting its abort flag.
    """
    counters[0] += 1
    if counters[0] > node_limit:
        counters[1] = 1
        return 0

    if halfmove_clock >= 100:
        return 0

    checked = in_check(board, side, offsets, n_offsets, is_slider)
    if not checked:
        stand = (
            evaluate_accumulator(nnue_acc, side, nnue_out, nnue_out_bias)
            if use_nnue else evaluate(board, side, pst, king_mid, king_end)
        )
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

    buf = np.empty(256, dtype=np.int32)
    n = gen_moves(board, side, castling, ep, buf, offsets, n_offsets, is_slider)
    scores = np.empty(n, dtype=np.int32)
    cnt = 0
    for i in range(n):
        m = buf[i]
        victim = board[mv_to(m)]
        # In check every legal move is an evasion candidate.  Otherwise retain
        # captures, promotions, en-passant, plus one layer of quiet checks.
        # The latter closes a common horizon: a quiet forcing check followed by
        # a capture was invisible to capture-only quiescence.  Restricting it to
        # qply zero prevents unbounded checking trees.
        noisy = (victim != EMPTY and victim != OFF) or mv_promo(m) != 0 or mv_flags(m) == 1
        if checked or noisy or qply == 0:
            if (
                not checked
                and not noisy
                and see(board, side, castling, ep, m, offsets, n_offsets, is_slider) < 0
            ):
                continue
            buf[cnt] = m
            scores[cnt] = _mvv_lva(board, m)
            cnt += 1

    legal = 0
    for i in range(cnt):
        best = i
        for j in range(i + 1, cnt):
            if scores[j] > scores[best]:
                best = j
        if best != i:
            scores[i], scores[best] = scores[best], scores[i]
            buf[i], buf[best] = buf[best], buf[i]

        m = buf[i]
        mover = board[mv_from(m)]
        captured, new_cr, new_ep = make_move(board, side, castling, ep, m)
        placed = board[mv_to(m)]
        if use_nnue:
            apply_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
        if in_check(board, side, offsets, n_offsets, is_slider):
            if use_nnue:
                undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
            unmake_move(board, side, m, captured)
            continue
        # A quiet move was admitted only to test whether it is a forcing check.
        quiet_noncheck = (
            not checked
            and not noisy
            and not in_check(board, 1 - side, offsets, n_offsets, is_slider)
        )
        if quiet_noncheck:
            if use_nnue:
                undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
            unmake_move(board, side, m, captured)
            continue
        legal += 1
        score = -quiesce(board, 1 - side, new_cr, new_ep, -beta, -alpha,
                         offsets, n_offsets, is_slider, pst, king_mid, king_end,
                         counters, node_limit,
                         next_halfmove_clock(mover, captured, halfmove_clock),
                         qply + 1,
                         nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
        if use_nnue:
            undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
        unmake_move(board, side, m, captured)
        if counters[1] == 1:
            return 0
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    if checked and legal == 0:
        return -MATE
    return alpha


@njit(cache=False)
def negamax(
    board: "np.ndarray", side: int, castling: int, ep: int, h: int, depth: int,
    alpha: int, beta: int, ply: int,
    offsets: "np.ndarray", n_offsets: "np.ndarray", is_slider: "np.ndarray",
    pst: "np.ndarray", king_mid: "np.ndarray", king_end: "np.ndarray",
    tt_key: "np.ndarray", tt_score: "np.ndarray", tt_move_a: "np.ndarray",
    tt_depth: "np.ndarray", tt_flag: "np.ndarray",
    killers: "np.ndarray", history: "np.ndarray", counters: "np.ndarray",
    node_limit: int,
    zp: "np.ndarray", zs: int, zc: "np.ndarray", ze: "np.ndarray",
    game_hashes: "np.ndarray", game_count: int,
    path_hashes: "np.ndarray", path_count: int,
    halfmove_clock: int,
    nnue_acc: "np.ndarray", nnue_ft: "np.ndarray", nnue_out: "np.ndarray",
    nnue_out_bias: int, use_nnue: bool,
) -> int:
    """Alpha-beta with a transposition table, killers, history and quiescence."""
    counters[0] += 1
    if counters[0] > node_limit:
        counters[1] = 1  # aborted
        return 0

    if halfmove_clock >= 100:
        return 0

    checked = in_check(board, side, offsets, n_offsets, is_slider)
    if checked:
        depth += 1  # check extension: never let a forcing line fall off the horizon

    if _is_third_repetition(h, game_hashes, game_count, path_hashes, path_count):
        return 0

    if depth <= 0:
        return quiesce(board, side, castling, ep, alpha, beta,
                       offsets, n_offsets, is_slider, pst, king_mid, king_end,
                       counters, node_limit, halfmove_clock, 0,
                       nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)

    next_path_count = path_count
    if path_count < MAX_PLY:
        path_hashes[path_count] = h
        next_path_count += 1
    tt_h = h ^ int(ZOB_HALF[halfmove_clock])
    idx = tt_h & TT_MASK
    tt_hit_move = 0
    if tt_key[idx] == tt_h:
        tt_hit_move = tt_move_a[idx]
        if tt_depth[idx] >= depth:
            f = tt_flag[idx]
            s = _score_from_tt(tt_score[idx], ply)
            if f == TT_EXACT:
                return int(s)
            if f == TT_LOWER and s > alpha:
                alpha = s
            elif f == TT_UPPER and s < beta:
                beta = s
            if alpha >= beta:
                return int(s)

    # Null-move pruning. If giving the opponent a free move still fails high, the
    # position is so good we can prune. Research (chessprogramming.org) is explicit
    # about the two guards that keep this sound: never when in check, and never in
    # likely zugzwang - endgames with no non-pawn material for the side to move,
    # where passing is artificially good. Also require a non-shallow depth and that
    # we are not already above beta on the static eval side.
    if (
        not checked
        and depth >= 3
        and beta < MATE - MAX_PLY
        and _has_non_pawn_material(board, side)
    ):
        # A null move: same board, opponent to move, en passant cleared. Search it
        # reduced by R and with a null window around beta.
        r = 2 + (depth // 6)
        null_hash = zobrist_after_null(h, ep, zs, ze)
        null_score = -negamax(board, 1 - side, castling, 0, null_hash, depth - 1 - r,
                              -beta, -beta + 1, ply + 1,
                              offsets, n_offsets, is_slider, pst, king_mid, king_end,
                              tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                              killers, history, counters, node_limit, zp, zs, zc, ze,
                              game_hashes, game_count, path_hashes, next_path_count,
                              halfmove_clock + 1,
                              nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
        if counters[1] == 1:
            return 0
        if null_score >= beta:
            return beta

    buf = np.empty(256, dtype=np.int32)
    n = gen_moves(board, side, castling, ep, buf, offsets, n_offsets, is_slider)
    scores = np.empty(n, dtype=np.int32)
    _order(
        board, buf, n, scores, tt_hit_move, killers, ply, history, side,
        castling, ep, offsets, n_offsets, is_slider,
    )

    best_score = -INF
    best_move = 0
    legal = 0
    alpha_orig = alpha

    for i in range(n):
        pick = i
        for j in range(i + 1, n):
            if scores[j] > scores[pick]:
                pick = j
        if pick != i:
            scores[i], scores[pick] = scores[pick], scores[i]
            buf[i], buf[pick] = buf[pick], buf[i]

        m = buf[i]
        mover = board[mv_from(m)]
        captured, new_cr, new_ep = make_move(board, side, castling, ep, m)
        placed = board[mv_to(m)]
        if use_nnue:
            apply_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
        child_halfmove_clock = next_halfmove_clock(mover, captured, halfmove_clock)
        child_hash = zobrist_after_move(
            h, board, side, castling, ep, m, captured, new_cr, new_ep, zp, zs, zc, ze
        )
        if in_check(board, side, offsets, n_offsets, is_slider):
            if use_nnue:
                undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
            unmake_move(board, side, m, captured)
            continue
        legal += 1

        is_quiet = (captured == EMPTY or captured == OFF) and mv_promo(m) == 0  # noqa: SIM109 (explicit compares; tuple `in` is slower under numba)
        # Late move reduction: quiet moves late in a well-ordered list rarely
        # deserve full depth. Verified with a re-search if the reduction fails high.
        red = late_move_reduction(depth, legal) if is_quiet and not checked else 0

        # Principal Variation Search. The first legal move is searched with the
        # full window. Every later move is first probed with a null window
        # [alpha, alpha+1] to prove cheaply that it is not better than what we
        # already have; only if that probe beats alpha (and there is real window
        # left) do we re-search it in full. Composes with LMR: the reduced null
        # window is the cheapest possible probe.
        if legal == 1:
            score = -negamax(board, 1 - side, new_cr, new_ep, child_hash, depth - 1,
                             -beta, -alpha, ply + 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze,
                             game_hashes, game_count, path_hashes, next_path_count,
                             child_halfmove_clock,
                             nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
        else:
            score = -negamax(board, 1 - side, new_cr, new_ep, child_hash, depth - 1 - red,
                             -alpha - 1, -alpha, ply + 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze,
                             game_hashes, game_count, path_hashes, next_path_count,
                             child_halfmove_clock,
                             nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
            # Failed high on the null window (or the reduction was too aggressive):
            # re-search with the full window at full depth.
            if score > alpha and (score < beta or red > 0):
                score = -negamax(board, 1 - side, new_cr, new_ep, child_hash, depth - 1,
                                 -beta, -alpha, ply + 1,
                                 offsets, n_offsets, is_slider, pst, king_mid,
                                 king_end, tt_key, tt_score, tt_move_a, tt_depth,
                                 tt_flag, killers, history, counters, node_limit,
                                 zp, zs, zc, ze,
                                 game_hashes, game_count, path_hashes, next_path_count,
                                 child_halfmove_clock,
                                 nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
        if use_nnue:
            undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
        unmake_move(board, side, m, captured)

        if counters[1] == 1:
            return 0

        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if is_quiet and ply < MAX_PLY:
                if killers[ply, 0] != m:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = m
                history[side, mv_from(m), mv_to(m)] += depth * depth
            break

    if legal == 0:
        # No legal move: mate if in check, otherwise stalemate. Encoding the ply
        # makes the engine prefer the fastest mate and the slowest loss.
        return -MATE + ply if checked else 0

    flag = TT_EXACT
    if best_score <= alpha_orig:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    if tt_depth[idx] <= depth or tt_key[idx] != tt_h:
        tt_key[idx] = tt_h
        tt_score[idx] = _score_to_tt(best_score, ply)
        tt_move_a[idx] = best_move
        tt_depth[idx] = depth
        tt_flag[idx] = flag

    return best_score


@njit(cache=False)
def search_root(board: "np.ndarray", side: int, castling: int, ep: int, depth: int,
                offsets: "np.ndarray", n_offsets: "np.ndarray",
                is_slider: "np.ndarray", pst: "np.ndarray",
                king_mid: "np.ndarray", king_end: "np.ndarray",
                tt_key: "np.ndarray", tt_score: "np.ndarray",
                tt_move_a: "np.ndarray", tt_depth: "np.ndarray",
                tt_flag: "np.ndarray", killers: "np.ndarray",
                history: "np.ndarray", counters: "np.ndarray",
                node_limit: int, zp: "np.ndarray", zs: int,
                zc: "np.ndarray", ze: "np.ndarray",
                prev_best: int, game_hashes: "np.ndarray",
                game_count: int, root_hash: int,
                halfmove_clock: int,
                nnue_acc: "np.ndarray", nnue_ft: "np.ndarray",
                nnue_out: "np.ndarray", nnue_out_bias: int,
                use_nnue: bool) -> "tuple[int, int]":
    """One iteration of iterative deepening. Returns (score, best_move)."""
    if halfmove_clock >= 100:
        return 0, 0
    buf = np.empty(256, dtype=np.int32)
    n = gen_moves(board, side, castling, ep, buf, offsets, n_offsets, is_slider)
    scores = np.empty(n, dtype=np.int32)
    _order(
        board, buf, n, scores, prev_best, killers, 0, history, side,
        castling, ep, offsets, n_offsets, is_slider,
    )

    alpha = -INF
    beta = INF
    best_move = 0
    best_score = -INF
    legal = 0
    catastrophic_fallback = 0
    root_checked = in_check(board, side, offsets, n_offsets, is_slider)
    path_hashes = np.zeros(MAX_PLY, dtype=np.int64)

    for i in range(n):
        pick = i
        for j in range(i + 1, n):
            if scores[j] > scores[pick]:
                pick = j
        if pick != i:
            scores[i], scores[pick] = scores[pick], scores[i]
            buf[i], buf[pick] = buf[pick], buf[i]

        m = buf[i]
        # SEE operates on the position before the capture.
        root_see = 0
        victim = board[mv_to(m)]
        if (
            not root_checked
            and mv_promo(m) == 0
            and ((victim != EMPTY and victim != OFF) or mv_flags(m) == 1)
        ):
            root_see = see(
                board, side, castling, ep, m, offsets, n_offsets, is_slider
            )
        mover = board[mv_from(m)]
        captured, new_cr, new_ep = make_move(board, side, castling, ep, m)
        placed = board[mv_to(m)]
        if use_nnue:
            apply_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
        child_halfmove_clock = next_halfmove_clock(mover, captured, halfmove_clock)
        child_hash = zobrist_after_move(
            root_hash, board, side, castling, ep, m, captured, new_cr, new_ep,
            zp, zs, zc, ze,
        )
        if in_check(board, side, offsets, n_offsets, is_slider):
            if use_nnue:
                undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
            unmake_move(board, side, m, captured)
            continue
        # Keep the original root search intact except for an immediately
        # catastrophic, non-checking capture.  This catches a queen simply
        # being taken (the rated Qxc4 loss was -570 SEE), not a normal exchange
        # or speculative piece sacrifice.
        catastrophic_capture = (
            not root_checked
            and captured != EMPTY
            and captured != OFF
            and mv_promo(m) == 0
            and not in_check(board, 1 - side, offsets, n_offsets, is_slider)
            and root_see <= -ROOT_CATASTROPHIC_SEE
        )
        if catastrophic_capture:
            if catastrophic_fallback == 0:
                catastrophic_fallback = m
            if use_nnue:
                undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
            unmake_move(board, side, m, captured)
            continue

        legal += 1
        if legal == 1:
            # The first root move establishes the principal variation.  As at
            # interior nodes, later moves get a cheap null-window probe first;
            # a probe that improves alpha is re-searched with the full window.
            score = -negamax(board, 1 - side, new_cr, new_ep, child_hash, depth - 1,
                             -beta, -alpha, 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze,
                             game_hashes, game_count, path_hashes, 0,
                             child_halfmove_clock,
                             nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
        else:
            score = -negamax(board, 1 - side, new_cr, new_ep, child_hash, depth - 1,
                             -alpha - 1, -alpha, 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze,
                             game_hashes, game_count, path_hashes, 0,
                             child_halfmove_clock,
                             nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
            if score > alpha and score < beta:
                score = -negamax(board, 1 - side, new_cr, new_ep, child_hash, depth - 1,
                                 -beta, -alpha, 1,
                                 offsets, n_offsets, is_slider, pst, king_mid, king_end,
                                 tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                                 killers, history, counters, node_limit, zp, zs, zc, ze,
                                 game_hashes, game_count, path_hashes, 0,
                                 child_halfmove_clock,
                                 nnue_acc, nnue_ft, nnue_out, nnue_out_bias, use_nnue)
        if use_nnue:
            undo_move(nnue_acc, mover, placed, captured, side, m, nnue_ft)
        unmake_move(board, side, m, captured)

        if counters[1] == 1:
            break
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score

    if legal == 0 and catastrophic_fallback != 0:
        return -INF, catastrophic_fallback
    return best_score, best_move

# ==========================================================================
# python-chess bridge and time-managed entry point
# ==========================================================================

_PIECE_TO_CODE = {
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9, (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
}
_SQ_NAME: dict[int, str] = {int(SQ120[s]): chess.square_name(s) for s in range(64)}
_CODE_TO_PROMO = {2: "n", 3: "b", 4: "r", 5: "q", 8: "n", 9: "b", 10: "r", 11: "q"}


def _to_arrays(board: chess.Board) -> tuple[np.ndarray, int, int, int]:
    """FEN-parsed board to the engine's arrays. Once per move, so cost is irrelevant."""
    arr = np.full(120, OFF, dtype=np.int8)
    for s in range(64):
        arr[int(SQ120[s])] = EMPTY
    for sq, piece in board.piece_map().items():
        arr[int(SQ120[sq])] = _PIECE_TO_CODE[(piece.piece_type, piece.color)]
    side = 0 if board.turn == chess.WHITE else 1
    cr = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        cr |= CR_WK
    if board.has_queenside_castling_rights(chess.WHITE):
        cr |= CR_WQ
    if board.has_kingside_castling_rights(chess.BLACK):
        cr |= CR_BK
    if board.has_queenside_castling_rights(chess.BLACK):
        cr |= CR_BQ
    ep = int(SQ120[board.ep_square]) if board.ep_square is not None else 0
    return arr, side, cr, ep


def _move_to_uci(m: int) -> str:
    uci = _SQ_NAME[mv_from(m)] + _SQ_NAME[mv_to(m)]
    promo = mv_promo(m)
    return uci + _CODE_TO_PROMO[promo] if promo else uci


class _Engine:
    """Holds the transposition table and ordering heuristics.

    These persist across our moves within a game, which IDEAS.md calls a real gain,
    and die with the process so they never leak into the next game.
    """

    def __init__(self) -> None:
        self.tt = new_tt()
        self.killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
        self.history = np.zeros((2, 120, 120), dtype=np.int32)
        # Root positions supplied by the runner. The process survives for one
        # game, so these are the real game history used to recognise a third
        # repetition in a searched variation.
        self.game_hashes = np.zeros(600, dtype=np.int64)
        self.game_count = 0
        # Measured node rate, refined after every real search so the node budget
        # tracks the machine we are actually running on rather than a guess.
        self.nps = 1_500_000.0
        # This is the team's own pre-event trained integer NNUE, ported from
        # the public C++ implementation. It remains off by default: a 20-game
        # A/B run did not show a strength gain over the established evaluator.
        # Keeping the path wired lets a retrained network be tested without a
        # risky search rewrite.
        self.use_nnue = False
        self.nnue_acc = np.zeros((2, 512), dtype=np.int32)
        self.nnue_ft = np.zeros((768, 512), dtype=np.int16)
        self.nnue_bias = np.zeros(512, dtype=np.int16)
        self.nnue_out = np.zeros(1024, dtype=np.int16)
        self.nnue_out_bias = 0
        if self.use_nnue:
            network_path = Path(__file__).resolve().parent / "weights" / "cpp_nnue.bin"
            nnue_ft, nnue_bias, nnue_out, nnue_out_bias = load_network(network_path)
            self.nnue_ft = nnue_ft
            self.nnue_bias = nnue_bias
            self.nnue_out = nnue_out
            self.nnue_out_bias = int(nnue_out_bias)

    def search(self, board: chess.Board, soft_s: float, hard_s: float) -> int:
        arr, side, cr, ep = _to_arrays(board)
        root_hash = zobrist(arr, side, cr, ep, ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP)
        if self.game_count < len(self.game_hashes):
            self.game_hashes[self.game_count] = root_hash
            self.game_count += 1
        # Decay history between moves: cutoffs from an earlier phase should inform
        # ordering, not dominate it.
        self.history //= 2
        best = 0
        start = time.time()
        nnue_acc = (
            refresh_accumulator(arr, self.nnue_ft, self.nnue_bias)
            if self.use_nnue else self.nnue_acc
        )

        for depth in range(1, MAX_PLY):
            elapsed = time.time() - start
            if elapsed >= soft_s:
                break
            # Do not START a depth that cannot plausibly finish inside the wall.
            # Each ply costs roughly 3x the last, so if the previous depth already
            # used more than a third of what is left, stop here rather than get
            # aborted mid-depth and waste the work.
            budget_left = hard_s - elapsed
            node_limit = int(self.nps * budget_left) + 20_000
            counters = np.zeros(2, dtype=np.int64)
            score, mv = search_root(
                arr, side, cr, ep, depth,
                OFFSETS, N_OFFSETS, IS_SLIDER, PST, PST_KING_MID, PST_KING_END,
                self.tt[0], self.tt[1], self.tt[2], self.tt[3], self.tt[4],
                self.killers, self.history, counters, node_limit,
                ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP, best,
                self.game_hashes, self.game_count, root_hash, board.halfmove_clock,
                nnue_acc, self.nnue_ft, self.nnue_out, int(self.nnue_out_bias),
                self.use_nnue,
            )
            spent = time.time() - start
            if spent > 0.02 and counters[0] > 0:
                measured = counters[0] / spent
                self.nps = 0.5 * self.nps + 0.5 * measured
            if counters[1] == 1:
                # Aborted inside this depth: its result is not trustworthy, so keep
                # the last completed depth's move.
                break
            if mv != 0:
                best = mv
            if score >= MATE - MAX_PLY or score <= -MATE + MAX_PLY:
                break  # proven mate, deeper search cannot improve on it
        return best


_ENGINE = _Engine()

# Milliseconds shaved off the clock for reply and accounting lag. The referee
# measures wall time and the watchdog does not forgive.
DELAY_MS = 200


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in ``fen``."""
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return ""
    if len(legal) == 1:
        return legal[0].uci()  # nothing to decide, and it banks the increment

    # Budget the clock across the moves we expect to still play. `soft` is when to
    # stop starting new iterations; `hard` bounds a single iteration. Both are
    # fractions of what remains, so the budget shrinks with the clock and can never
    # drive it to zero - a flag is an automatic loss and the easiest one to inflict
    # on yourself.
    remaining = max(time_left_ms - DELAY_MS, 0)
    moves_left = max(20, 60 - board.fullmove_number)
    budget_ms = remaining / moves_left
    soft_s = max(min(budget_ms, remaining / 6), 20) / 1000
    hard_s = max(min(2.0 * budget_ms, remaining / 8), 40) / 1000

    try:
        best = _ENGINE.search(board, soft_s, hard_s)
    except Exception:
        # Never take the process down; a legal move always beats a crash.
        return legal[0].uci()

    if best != 0:
        uci = _move_to_uci(best)
        try:
            if chess.Move.from_uci(uci) in board.legal_moves:
                return uci
        except ValueError:
            pass
    return legal[0].uci()


def _warmup() -> None:
    """Compile every jitted function at import, with the argument types the real
    calls use, so compilation lands in the 60 second init budget."""
    with contextlib.suppress(Exception):
        # Refreshing the 2x512 accumulator itself compiles on the first call.
        # A 50 ms soft budget expires before ``search_root`` gets invoked, which
        # used to leave its seven-second Numba compilation on our first move.
        # This generous soft budget lets the real recursive signature compile;
        # the one-second hard budget stops immediately after that compilation.
        _ENGINE.search(chess.Board(), 20.0, 1.0)
    # Discard anything the warmup learned so the first real move starts clean.
    _ENGINE.tt = new_tt()
    _ENGINE.history[:] = 0
    _ENGINE.killers[:] = 0
    _ENGINE.game_hashes[:] = 0
    _ENGINE.game_count = 0
    _ENGINE.nps = 1_500_000.0


_warmup()
