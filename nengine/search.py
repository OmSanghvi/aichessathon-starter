"""Jitted negamax search: alpha-beta, transposition table, quiescence, ordering.

Built on the perft-verified generator in nengine/board.py, which runs at 15-22M
nodes a second against the pure-Python engine's 55,000. IDEAS.md prescribes exactly
this shape - "negamax with alpha-beta is the whole game", ordering first, a
transposition table, iterative deepening, and quiescence with captures only - and
says the payoff is the depth the speed buys.

DESIGN NOTES

Time is enforced with a NODE BUDGET rather than a clock read inside the jitted
search. The Python driver measures how many nodes a second it is actually getting,
converts the remaining time into a budget, and runs iterative deepening; each depth
aborts cleanly when the budget is spent. That keeps the hot loop free of any call
back into Python and makes a search reproducible, which matters because a
non-deterministic search cannot be A/B tested honestly.

The transposition table is open-addressed numpy arrays rather than a dict: numba's
typed dicts are slow, and a fixed-size table with index = hash & mask costs one
memory access. It survives across moves within a game, which IDEAS.md calls a real
gain, and is only cleared when the position changes incompatibly.

Ordering, in the order it matters: transposition-table move, then captures by
MVV-LVA, then killers, then history. That is the standard recipe and it is what
makes alpha-beta pay.
"""

import numpy as np
from numba import njit

from nengine.board import (
    EMPTY,
    OFF,
    gen_moves,
    in_check,
    make_move,
    mv_flags,
    mv_from,
    mv_promo,
    mv_to,
    unmake_move,
)

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


@njit(cache=False)
def evaluate(
    board: "np.ndarray",
    side: int,
    pst: "np.ndarray",
    king_mid: "np.ndarray",
    king_end: "np.ndarray",
) -> int:
    """Material plus piece-square, from the side-to-move's point of view."""
    score = 0
    npm_w = 0
    npm_b = 0
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
            kind = p - 6
            if kind != 1 and kind != 6:
                npm_b += PIECE_VALUE[kind]
            if kind != 6:
                score -= PIECE_VALUE[kind] + pst[kind, s ^ 56]

    endgame = (npm_w + npm_b) <= ENDGAME_MATERIAL
    for s in range(64):
        sq = 91 + (s & 7) - 10 * (s >> 3)
        p = board[sq]
        if p == 6:
            score += king_end[s] if endgame else king_mid[s]
        elif p == 12:
            score -= king_end[s ^ 56] if endgame else king_mid[s ^ 56]

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
        score = -negamax(board, 1 - side, new_cr, new_ep, depth - 1,
                         -beta, -alpha, 1,
                         offsets, n_offsets, is_slider, pst, king_mid, king_end,
                         tt_key, tt_score, tt_move_a, tt_depth, tt_flag,
                         killers, history, counters, node_limit, zp, zs, zc, ze)
        unmake_move(board, side, m, captured)

        if counters[1] == 1:
            break
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score

    return best_score, best_move
