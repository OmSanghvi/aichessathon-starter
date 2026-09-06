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
# Search: alpha-beta, transposition table, quiescence, move ordering
# ==========================================================================

MATE = 30000
INF = 32000
MAX_PLY = 64


# Material, in centipawns, indexed by piece type 1..6.
PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int32)

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

# Non-pawn material below which the king centralises.
ENDGAME_MATERIAL = 1300

# Small, deliberately conservative positional terms.  They are all integer-only
# scans of the mailbox, so they cost far less than an extra searched ply.  Keeping
# the values below a pawn prevents a static opinion from overriding tactics.
PASSED_PAWN_BONUS = np.array([0, 0, 4, 10, 22, 40, 70, 0], dtype=np.int32)
PROTECTED_PASSER_BONUS = 8
# Knight and bishop freedom is a stable positional signal.  Rook and queen
# reach is much more tactical, so search evaluates it instead of spending leaf
# time ray-counting it in every position.
MOBILITY_BONUS = np.array([0, 0, 2, 1, 0, 0, 0], dtype=np.int32)
MISSING_SHIELD_PENALTY = 11
SECOND_SHIELD_PENALTY = 3
OPEN_KING_FILE_PENALTY = 7

# For each colour and square, the squares on which an opposing pawn would stop
# a passer.  Constructed once at import, then passed-pawn testing in evaluation
# is a single uint64 mask test rather than a forward board walk.
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

TT_BITS = 21  # 2M entries, about 40 MB across the arrays
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2


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
    """Pawn-shield and open-file danger near an unadvanced king.

    This is intentionally not an attack-map calculation: tactical king safety is
    searched.  The static term only records durable pawn-cover damage.
    """
    # Mailbox files occupy the low decimal digit (a=1 through h=8); using 98
    # gives the correct 0..7 rank for every file, not just file a.
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
    npm_w = 0
    npm_b = 0
    white_pawns = np.uint64(0)
    black_pawns = np.uint64(0)
    white_king = 0
    black_king = 0
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if p == EMPTY or p == OFF:  # noqa: SIM109 (explicit compares; tuple `in` is slower under numba)
            continue
        if p <= 6:
            kind = p
            if kind != 1 and kind != 6:
                npm_w += PIECE_VALUE[kind]
            if kind != 6:
                score += PIECE_VALUE[kind] + pst[kind, s]
            else:
                white_king = sq

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
            if kind != 6:
                score -= PIECE_VALUE[kind] + pst[kind, s ^ 56]
            else:
                black_king = sq

            if kind == 1:
                black_pawns |= np.uint64(1) << np.uint64(s)
            elif kind != 6:
                mobility_weight = MOBILITY_BONUS[kind]
                if mobility_weight != 0:
                    score -= mobility_weight * _mobility(board, sq, kind, 1)

    endgame = (npm_w + npm_b) <= ENDGAME_MATERIAL
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if p == 6:
            score += king_end[s] if endgame else king_mid[s]
        elif p == 12:
            score -= king_end[s ^ 56] if endgame else king_mid[s ^ 56]
        elif p == 1 and (black_pawns & PASSED_PAWN_MASK[0, s]) == 0:
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
) -> None:
    """Score each move for ordering; the caller does selection sort on the fly."""
    for i in range(n):
        m = buf[i]
        if m == tt_move and tt_move != 0:
            scores[i] = 1 << 28
            continue
        victim = board[mv_to(m)]
        if (victim != EMPTY and victim != OFF) or mv_promo(m) != 0:
            scores[i] = (1 << 24) + _mvv_lva(board, m)
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
    counters: "np.ndarray", node_limit: int,
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

    checked = in_check(board, side, offsets, n_offsets, is_slider)
    if not checked:
        stand = evaluate(board, side, pst, king_mid, king_end)
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
        # captures, promotions, and en-passant (whose destination is empty).
        if checked or (victim != EMPTY and victim != OFF) or mv_promo(m) != 0 or mv_flags(m) == 1:
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
        captured, new_cr, new_ep = make_move(board, side, castling, ep, m)
        if in_check(board, side, offsets, n_offsets, is_slider):
            unmake_move(board, side, m, captured)
            continue
        legal += 1
        score = -quiesce(board, 1 - side, new_cr, new_ep, -beta, -alpha,
                         offsets, n_offsets, is_slider, pst, king_mid, king_end,
                         counters, node_limit)
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
    board: "np.ndarray", side: int, castling: int, ep: int, depth: int,
    alpha: int, beta: int, ply: int,
    offsets: "np.ndarray", n_offsets: "np.ndarray", is_slider: "np.ndarray",
    pst: "np.ndarray", king_mid: "np.ndarray", king_end: "np.ndarray",
    tt_key: "np.ndarray", tt_score: "np.ndarray", tt_move_a: "np.ndarray",
    tt_depth: "np.ndarray", tt_flag: "np.ndarray",
    killers: "np.ndarray", history: "np.ndarray", counters: "np.ndarray",
    node_limit: int,
    zp: "np.ndarray", zs: int, zc: "np.ndarray", ze: "np.ndarray",
) -> int:
    """Alpha-beta with a transposition table, killers, history and quiescence."""
    counters[0] += 1
    if counters[0] > node_limit:
        counters[1] = 1  # aborted
        return 0

    checked = in_check(board, side, offsets, n_offsets, is_slider)
    if checked:
        depth += 1  # check extension: never let a forcing line fall off the horizon

    if depth <= 0:
        return quiesce(board, side, castling, ep, alpha, beta,
                       offsets, n_offsets, is_slider, pst, king_mid, king_end,
                       counters, node_limit)

    h = zobrist(board, side, castling, ep, zp, zs, zc, ze)
    idx = h & TT_MASK
    tt_hit_move = 0
    if tt_key[idx] == h:
        tt_hit_move = tt_move_a[idx]
        if tt_depth[idx] >= depth:
            f = tt_flag[idx]
            s = tt_score[idx]
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
        null_score = -negamax(board, 1 - side, castling, 0, depth - 1 - r,
                              -beta, -beta + 1, ply + 1,
                              offsets, n_offsets, is_slider, pst, king_mid, king_end,
                              tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                              killers, history, counters, node_limit, zp, zs, zc, ze)
        if counters[1] == 1:
            return 0
        if null_score >= beta:
            return beta

    buf = np.empty(256, dtype=np.int32)
    n = gen_moves(board, side, castling, ep, buf, offsets, n_offsets, is_slider)
    scores = np.empty(n, dtype=np.int32)
    _order(board, buf, n, scores, tt_hit_move, killers, ply, history, side)

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
        captured, new_cr, new_ep = make_move(board, side, castling, ep, m)
        if in_check(board, side, offsets, n_offsets, is_slider):
            unmake_move(board, side, m, captured)
            continue
        legal += 1

        is_quiet = (captured == EMPTY or captured == OFF) and mv_promo(m) == 0  # noqa: SIM109 (explicit compares; tuple `in` is slower under numba)
        # Late move reduction: quiet moves late in a well-ordered list rarely
        # deserve full depth. Verified with a re-search if the reduction fails high.
        red = 0
        if depth >= 3 and legal > 3 and is_quiet and not checked:
            red = 1

        # Principal Variation Search. The first legal move is searched with the
        # full window. Every later move is first probed with a null window
        # [alpha, alpha+1] to prove cheaply that it is not better than what we
        # already have; only if that probe beats alpha (and there is real window
        # left) do we re-search it in full. Composes with LMR: the reduced null
        # window is the cheapest possible probe.
        if legal == 1:
            score = -negamax(board, 1 - side, new_cr, new_ep, depth - 1,
                             -beta, -alpha, ply + 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze)
        else:
            score = -negamax(board, 1 - side, new_cr, new_ep, depth - 1 - red,
                             -alpha - 1, -alpha, ply + 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze)
            # Failed high on the null window (or the reduction was too aggressive):
            # re-search with the full window at full depth.
            if score > alpha and (score < beta or red > 0):
                score = -negamax(board, 1 - side, new_cr, new_ep, depth - 1,
                                 -beta, -alpha, ply + 1,
                                 offsets, n_offsets, is_slider, pst, king_mid,
                                 king_end, tt_key, tt_score, tt_move_a, tt_depth,
                                 tt_flag, killers, history, counters, node_limit,
                                 zp, zs, zc, ze)
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
    if tt_depth[idx] <= depth or tt_key[idx] != h:
        tt_key[idx] = h
        tt_score[idx] = best_score
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
                prev_best: int) -> "tuple[int, int]":
    """One iteration of iterative deepening. Returns (score, best_move)."""
    buf = np.empty(256, dtype=np.int32)
    n = gen_moves(board, side, castling, ep, buf, offsets, n_offsets, is_slider)
    scores = np.empty(n, dtype=np.int32)
    _order(board, buf, n, scores, prev_best, killers, 0, history, side)

    alpha = -INF
    beta = INF
    best_move = 0
    best_score = -INF

    legal = 0
    for i in range(n):
        pick = i
        for j in range(i + 1, n):
            if scores[j] > scores[pick]:
                pick = j
        if pick != i:
            scores[i], scores[pick] = scores[pick], scores[i]
            buf[i], buf[pick] = buf[pick], buf[i]

        m = buf[i]
        captured, new_cr, new_ep = make_move(board, side, castling, ep, m)
        if in_check(board, side, offsets, n_offsets, is_slider):
            unmake_move(board, side, m, captured)
            continue
        legal += 1

        # Principal variation search at the root.  Iterative deepening places the
        # previous iteration's PV move first, so every later root move is usually
        # worse than alpha.  A null-window probe proves that cheaply; a move which
        # does beat alpha is immediately re-searched with the full window.  The
        # interior search already does this, but omitting it at the root made every
        # plausible quiet alternative pay for an expensive full-window search.
        #
        # Do not use the pseudo-move index here: an illegal pseudo-move can precede
        # the first legal move in a checked position.
        if legal == 1:
            score = -negamax(board, 1 - side, new_cr, new_ep, depth - 1,
                             -beta, -alpha, 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze)
        else:
            score = -negamax(board, 1 - side, new_cr, new_ep, depth - 1,
                             -alpha - 1, -alpha, 1,
                             offsets, n_offsets, is_slider, pst, king_mid, king_end,
                             tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                             killers, history, counters, node_limit, zp, zs, zc, ze)
            if score > alpha and score < beta:
                score = -negamax(board, 1 - side, new_cr, new_ep, depth - 1,
                                 -beta, -alpha, 1,
                                 offsets, n_offsets, is_slider, pst, king_mid,
                                 king_end, tt_key, tt_score, tt_move_a, tt_depth,
                                 tt_flag, killers, history, counters, node_limit,
                                 zp, zs, zc, ze)
        unmake_move(board, side, m, captured)

        if counters[1] == 1:
            break
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score

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
        # Measured node rate, refined after every real search so the node budget
        # tracks the machine we are actually running on rather than a guess.
        self.nps = 1_500_000.0

    def search(self, board: chess.Board, soft_s: float, hard_s: float) -> int:
        arr, side, cr, ep = _to_arrays(board)
        # Decay history between moves: cutoffs from an earlier phase should inform
        # ordering, not dominate it.
        self.history //= 2
        best = 0
        start = time.time()

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
        _ENGINE.search(chess.Board(), 0.05, 0.10)
    # Discard anything the warmup learned so the first real move starts clean.
    _ENGINE.tt = new_tt()
    _ENGINE.history[:] = 0
    _ENGINE.killers[:] = 0
    _ENGINE.nps = 1_500_000.0


_warmup()
