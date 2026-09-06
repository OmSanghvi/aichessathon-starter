"""Numba-jitted board and move generator: the foundation of the fast engine.

WHY THIS EXISTS
The pure-Python engine runs at about 55,000 nodes a second, and profiling put 37%
of that in move generation and 16% in the incremental evaluation, all of it Python
string manipulation. IDEAS.md is explicit that numba is how Python gets fast here
and that "the gain is the depth the speed lets you afford". Four to six extra plies
is worth more than every evaluation and ordering tweak available, and depth is
exactly what the engine has been losing games to.

REPRESENTATION
A 10x12 mailbox, the same trick sunfish uses: the board is a 120-entry array with an
off-board sentinel ring, so a slider walking off the edge hits OFF and stops without
any bounds arithmetic. Squares 0..63 map to mailbox indices via SQ120.

  piece codes  0 empty, 1..6 white P N B R Q K, 7..12 black p n b r q k, OFF=13
  side         0 white, 1 black
  castling     4 bits: 1 white kingside, 2 white queenside, 4 black kingside,
               8 black queenside
  ep           mailbox index of the en-passant target, or 0

MOVE ENCODING
One int32 so moves live in plain numpy arrays rather than Python objects:

  bits 0-7    from (mailbox index)
  bits 8-15   to   (mailbox index)
  bits 16-19  promotion piece code, 0 for none
  bits 20-23  flags: 1 en-passant capture, 2 castling

Generation is PSEUDO-LEGAL; legality is decided by making the move and asking
whether our own king is attacked. That is slower than pin-aware generation but it is
far easier to get right, and correctness here is non-negotiable - perft against
python-chess is the gate.
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12
OFF = 13

N, E, S, W = -10, 1, 10, -1

# Castling right bits.
CR_WK, CR_WQ, CR_BK, CR_BQ = 1, 2, 4, 8

# Mailbox index for each 0..63 square. Square 0 is a1 at mailbox 91, and north
# (increasing rank) is -10, matching sunfish's layout.
SQ120 = np.array(
    [91 + (s & 7) - 10 * (s >> 3) for s in range(64)], dtype=np.int8
)
# Reverse map, -1 where the mailbox entry is not a real square.
SQ64 = np.full(120, -1, dtype=np.int8)
for _s in range(64):
    SQ64[SQ120[_s]] = _s

# Piece movement offsets, padded to a rectangular array so numba can index it.
# Row order: pawn, knight, bishop, rook, queen, king (1..6).
OFFSETS = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],  # pawn handled separately
        [-21, -19, -12, -8, 8, 12, 19, 21],  # knight
        [-11, -9, 9, 11, 0, 0, 0, 0],  # bishop
        [-10, -1, 1, 10, 0, 0, 0, 0],  # rook
        [-11, -10, -9, -1, 1, 9, 10, 11],  # queen
        [-11, -10, -9, -1, 1, 9, 10, 11],  # king
    ],
    dtype=np.int8,
)
N_OFFSETS = np.array([0, 8, 4, 4, 8, 8], dtype=np.int8)
IS_SLIDER = np.array([False, False, True, True, True, False], dtype=np.bool_)


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
