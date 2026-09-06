"""AI Chessathon submission entrypoint.

The platform imports this module once per game and calls get_move(fen, time_left_ms).
This agent embeds the sunfish engine (github.com/thomasahle/sunfish, GPLv3) and bridges
its 120-char board representation to python-chess so the platform's FEN-in / UCI-out
contract is satisfied.

Sunfish is a king-capture negamax engine driven by an MTD-bi iterative-deepening loop.
None of the engine internals depend on the network, threads, or a filesystem, so it runs
inside the platform's one-core / 2 GB / no-network sandbox unchanged.
"""

import time
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import NamedTuple

import chess
import numpy as np
from numba import njit

###############################################################################
# Learned evaluation (NNUE-style), replacing the static leaf evaluation only
###############################################################################
# A small net trained on positions labelled by a deep search. It substitutes for
# sunfish's stand-pat score at the search leaves; the incremental pos.score still
# drives move ordering and futility, because sunfish's tuned constants live in
# that centipawn scale and the net is trained to the same scale.
#
# Features are KING-BUCKETED (KING_BUCKETS x 12 x 64 = 3072): the net sees piece
# placement conditioned on which region its own king is in, which plain
# piece-square inputs cannot express. Orientation is sunfish's mover frame (a 180
# degree rotation for the side not moving), matching training exactly - the
# encoding here must agree with nnue2/features.py bit for bit or the net is fed
# inputs it never saw. nnue2/verify_features.py asserts the two agree.

NNUE_KING_BUCKETS = 4
NNUE_ACT_MAX = 127
NNUE_SHIFT = 6  # log2 of the hidden-layer weight scale
NNUE_W_SCALE = 64.0
NNUE_CP_SCALE = 400.0  # K_CP from training
NNUE_PIECE_ORDER = "PNBRQK"

# How far a residual net may move the evaluation away from pos.score. The search's
# tuned margins assume the leaf score lives near the piece-square score, so the
# correction is bounded to keep them meaningful while still letting the net express
# real positional knowledge.
#
# Swept against a reference engine: correlation with truth runs 0.719 for the tables
# alone, 0.763 at 300, peaks at 0.773 at 500, then flattens. Most of the anchoring
# comes from training on the RESIDUAL rather than from this clamp - even unclamped
# the residual net stays within ~300 cp of pos.score, where the absolute net strayed
# 591 - so the clamp only has to catch the tail.
NNUE_MAX_CORRECTION = 500


def _nnue_square_table() -> "np.ndarray":
    """Sunfish 120-board index -> 0..63 square, -1 for padding."""
    table = np.full(256, -1, dtype=np.int64)
    for i in range(120):
        f = (i - 91) % 10  # A1 == 91
        r = -((i - 91) // 10)
        if 0 <= f <= 7 and 0 <= r <= 7:
            table[i] = r * 8 + f
    return table


def _nnue_piece_table() -> "np.ndarray":
    """Char code -> relative_colour * 6 + piece_type, -1 when not a piece."""
    table = np.full(256, -1, dtype=np.int64)
    for t, ch in enumerate(NNUE_PIECE_ORDER):
        table[ord(ch)] = t  # uppercase: the side to move
        table[ord(ch.lower())] = 6 + t  # lowercase: the opponent
    return table


@njit(cache=False)
def _nnue_king_bucket(square: int) -> int:
    # Matches nnue2.features.king_bucket exactly.
    file_ = square & 7
    rank = square >> 3
    kingside = 1 if file_ >= 4 else 0
    advanced = 1 if rank >= 2 else 0
    return advanced * 2 + kingside


@njit(cache=False)
def _nnue_forward(
    codes: "np.ndarray",
    square_table: "np.ndarray",
    piece_table: "np.ndarray",
    ft_w: "np.ndarray",
    ft_b: "np.ndarray",
    l1_w: "np.ndarray",
    l1_b: "np.ndarray",
    l2_w: "np.ndarray",
    l2_b: "np.ndarray",
) -> int:
    acc_n = ft_b.shape[0]

    # The mover's king ('K', uppercase) sets the bucket for every feature.
    bucket = 0
    for i in range(codes.shape[0]):
        if codes[i] == 75:  # ord('K')
            sq = square_table[i]
            if sq >= 0:
                bucket = _nnue_king_bucket(sq)
            break
    base = bucket * 12

    acc = np.empty(acc_n, dtype=np.int32)
    for j in range(acc_n):
        acc[j] = ft_b[j]

    for i in range(codes.shape[0]):
        kind = piece_table[codes[i]]
        if kind < 0:
            continue
        square = square_table[i]
        if square < 0:
            continue
        feature = (base + kind) * 64 + square
        for j in range(acc_n):
            acc[j] += ft_w[feature, j]

    # Clipped ReLU; acc is already in ACT_MAX units, so saturate, do not divide.
    for j in range(acc_n):
        v = acc[j]
        if v < 0:
            acc[j] = 0
        elif v > NNUE_ACT_MAX:
            acc[j] = NNUE_ACT_MAX

    hidden = l1_w.shape[1]
    total = 0
    for k in range(hidden):
        s = l1_b[k]
        for j in range(acc_n):
            s += acc[j] * l1_w[j, k]
        s >>= NNUE_SHIFT
        if s < 0:
            s = 0
        elif s > NNUE_ACT_MAX:
            s = NNUE_ACT_MAX
        total += s * l2_w[k, 0]
    return int(total + l2_b[0])


class _Nnue:
    def __init__(self, path: Path) -> None:
        with np.load(str(path)) as z:
            self.ft_w = np.ascontiguousarray(z["ft_w"], dtype=np.int32)
            self.ft_b = np.ascontiguousarray(z["ft_b"], dtype=np.int32)
            self.l1_w = np.ascontiguousarray(z["l1_w"], dtype=np.int32)
            self.l1_b = np.ascontiguousarray(z["l1_b"], dtype=np.int32)
            self.l2_w = np.ascontiguousarray(z["l2_w"], dtype=np.int32)
            self.l2_b = np.ascontiguousarray(z["l2_b"], dtype=np.int32)
            # Residual nets output a CORRECTION to pos.score rather than an
            # absolute evaluation. Which one it is travels with the weights,
            # because guessing wrong silently discards or doubles the
            # piece-square term.
            self.residual = bool(int(z["residual"][0])) if "residual" in z else False
        self.square_table = _nnue_square_table()
        self.piece_table = _nnue_piece_table()
        self.scale = NNUE_W_SCALE * NNUE_ACT_MAX

    def evaluate(self, board: str) -> int:
        codes = np.frombuffer(board.encode("latin-1"), dtype=np.uint8).astype(np.int64)
        raw = _nnue_forward(
            codes, self.square_table, self.piece_table,
            self.ft_w, self.ft_b, self.l1_w, self.l1_b, self.l2_w, self.l2_b,
        )
        # round(), not int(): int() truncates toward zero and disagrees with the
        # training reference by a centipawn near .5 boundaries.
        return round(raw / self.scale * NNUE_CP_SCALE)


def _load_nnue() -> "_Nnue | None":
    """Load the net, or None so the search falls back to the piece-square score.

    A missing or broken net must never take the process down: playing on with the
    tables is far better than losing the game to an import error.
    """
    path = Path(__file__).resolve().parent / "weights" / "nnue.npz"
    if not path.exists():
        return None
    try:
        net = _Nnue(path)
        net.evaluate("." * 120)  # compile the kernel inside the import budget
        return net
    except Exception:
        return None


_NNUE = _load_nnue()

###############################################################################
# Piece-square tables (sunfish)
###############################################################################

piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst: dict[str, tuple[int, ...]] = {
    "P": (0, 0, 0, 0, 0, 0, 0, 0,
          78, 83, 86, 73, 102, 82, 85, 90,
          7, 29, 21, 44, 40, 31, 44, 7,
          -17, 16, -2, 15, 14, 0, 15, -13,
          -26, 3, 10, 9, 6, 1, 0, -23,
          -22, 9, 5, -11, -10, -2, 3, -19,
          -31, 8, -7, -37, -36, -14, 3, -31,
          0, 0, 0, 0, 0, 0, 0, 0),
    "N": (-66, -53, -75, -75, -10, -55, -58, -70,
          -3, -6, 100, -36, 4, 62, -4, -14,
          10, 67, 1, 74, 73, 27, 62, -2,
          24, 24, 45, 37, 33, 41, 25, 17,
          -1, 5, 31, 21, 22, 35, 2, 0,
          -18, 10, 13, 22, 18, 15, 11, -14,
          -23, -15, 2, 0, 2, 0, -23, -20,
          -74, -23, -26, -24, -19, -35, -22, -69),
    "B": (-59, -78, -82, -76, -23, -107, -37, -50,
          -11, 20, 35, -42, -39, 31, 2, -22,
          -9, 39, -32, 41, 52, -10, 28, -14,
          25, 17, 20, 34, 26, 25, 15, 10,
          13, 10, 17, 23, 17, 16, 0, 7,
          14, 25, 24, 15, 8, 25, 20, 15,
          19, 20, 11, 6, 7, 6, 20, 16,
          -7, 2, -15, -12, -14, -15, -10, -10),
    "R": (35, 29, 33, 4, 37, 33, 56, 50,
          55, 29, 56, 67, 55, 62, 34, 60,
          19, 35, 28, 33, 45, 27, 25, 15,
          0, 5, 16, 13, 18, -4, -9, -6,
          -28, -35, -16, -21, -13, -29, -46, -30,
          -42, -28, -42, -25, -25, -35, -26, -46,
          -53, -38, -31, -26, -29, -43, -44, -53,
          -30, -24, -18, 5, -2, -18, -31, -32),
    "Q": (6, 1, -8, -104, 69, 24, 88, 26,
          14, 32, 60, -10, 20, 76, 57, 24,
          -2, 43, 32, 60, 72, 63, 43, 2,
          1, -16, 22, 17, 25, 20, -13, -6,
          -14, -15, -2, -5, -1, -10, -20, -22,
          -30, -6, -13, -11, -16, -11, -16, -27,
          -36, -18, 0, -19, -15, -15, -21, -38,
          -39, -30, -31, -13, -31, -36, -34, -42),
    "K": (4, 54, 47, -99, -99, 60, 83, -62,
          -32, 10, 55, 56, 56, 55, 10, 3,
          -62, 12, -57, 44, -67, 28, 37, -31,
          -55, 50, 11, -4, -19, 13, 0, -49,
          -55, -43, -52, -28, -51, -47, -8, -50,
          -47, -42, -43, -79, -64, -32, -29, -32,
          -4, 3, -14, -50, -57, -18, 13, 4,
          17, 30, -3, -14, 6, -1, 40, 18),
}


# Pad tables to the 120-char board and fold in the base piece value.
def _pad_table(key: str, table: tuple[int, ...]) -> tuple[int, ...]:
    base = piece[key]
    padded = sum(((0, *(x + base for x in table[i * 8:i * 8 + 8]), 0) for i in range(8)), ())
    return (0,) * 20 + padded + (0,) * 20


pst = {k: _pad_table(k, t) for k, t in pst.items()}

# King table for the endgame: centralisation, needed to win KRK / KQK.
K_MID = pst["K"]
K_END = tuple(
    piece["K"] + 70 - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120)
)

###############################################################################
# Global constants (sunfish)
###############################################################################

A1, H1, A8, H8 = 91, 98, 21, 28
N, E, S, W = -10, 1, 10, -1

# Precomputed pawn offsets. These are only spelled out because N is a module
# global, so an expression like `d in (N, N + N)` in the move generator rebuilt a
# tuple on every iteration of its innermost loop.
NN = N + N
NW = N + W
NE = N + E
A1_N = A1 + N
# Pre-built so the generator's inner loop tests membership against an existing
# tuple instead of constructing one from globals on every iteration.
PAWN_PUSH = (N, NN)
PAWN_CAPTURE = (NW, NE)
directions = {
    "P": (N, N + N, N + W, N + E),
    "N": (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    "B": (N + E, S + E, S + W, N + W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N + E, S + E, S + W, N + W),
    "K": (N, E, S, W, N + E, S + E, S + W, N + W),
}

MATE_LOWER = piece["K"] - 13 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]
QS = 36
QS_A = 180
LMR = 70
EVAL_ROUGHNESS = 15
NULL_MARGIN = -200
TABLE_SIZE = 10**6

class Move(NamedTuple):
    i: int
    j: int
    prom: str


class Position(NamedTuple):
    """A chess position in sunfish's 120-char representation, always seen from
    the side to move."""

    board: str
    score: int
    wc: tuple[bool, bool]
    bc: tuple[bool, bool]
    ep: int
    kp: int

    def gen_moves(self) -> "Iterator[Move]":
        # This is the hottest function in the engine: profiling put it at 37% of
        # all search time across 6.6M calls. Everything below is the same logic
        # as the reference, rewritten to keep the inner loop cheap:
        #   * fields are hoisted to locals; `self.board[j]` in the innermost loop
        #     is a namedtuple attribute lookup on every iteration.
        #   * `d in (N, N + N)` built a fresh tuple from globals each time round,
        #     because N is a module global rather than a literal. Precomputed
        #     NN/NW/NE and plain comparisons avoid that.
        #   * `abs()` was called 9.1M times. kp is 0 except just after a castle,
        #     and every target square is >= A8, so abs(j - 0) > 1 always holds;
        #     testing kp first short-circuits almost all of those calls.
        board = self.board
        ep = self.ep
        kp = self.kp
        wc = self.wc
        for i, p in enumerate(board):
            if p not in "PNBRQK":
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = board[j]
                    if q in " \nPNBRQK":
                        break
                    if p == "P":
                        # The `d in (...)` form reads better but measured slower:
                        # tuple membership costs more than two integer compares,
                        # and this line runs millions of times per search.
                        if (d == N or d == NN) and q != ".":  # noqa: SIM109
                            break
                        if d == NN and (i < A1_N or board[i + N] != "."):
                            break
                        if ((d == NW or d == NE) and q == "."  # noqa: SIM109
                                and j != ep and (not kp or abs(j - kp) > 1)):
                            break
                        if A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                            break
                    yield Move(i, j, "")
                    if p in "PNK" or q in "pnbrqk":
                        break
                    if i == A1 and board[j + E] == "K" and wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and board[j + W] == "K" and wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove: bool = False) -> "Position":
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )

    def move(self, move: "Move") -> "Position":
        i, j, prom = move
        def put(board: str, at: int, p: str) -> str:
            return board[:at] + p + board[at + 1:]

        p, board, wc, bc, ep, kp = self.board[i], self.board, self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        board = put(board, j, board[i])
        board = put(board, i, ".")
        wc = (wc[0] and i != A1, wc[1] and i != H1)
        bc = (bc[0] and j != H8, bc[1] and j != A8)
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move: "Move") -> int:
        # Second hottest function: 16% of search time over 5.9M calls. Same
        # transformations as gen_moves - hoist the table lookup, and test kp
        # before paying for abs(), which cannot succeed while kp is 0 because
        # every target square is >= A8.
        i, j, prom = move
        board = self.board
        p, q = board[i], board[j]
        pst_p = pst[p]
        score = pst_p[j] - pst_p[i]
        if q in "pnbrqk":
            score += pst[q.upper()][119 - j]
        kp = self.kp
        if kp and abs(j - kp) < 2:
            score += pst["K"][119 - j]
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j + S)]
        return score

    def king_capture(self) -> "Move | None":
        return next(
            (m for m in self.gen_moves() if self.board[m.j] == "k" or abs(m.j - self.kp) < 2), None
        )


class Stop(Exception):
    pass


class Entry(NamedTuple):
    lower: int
    upper: int


class Searcher:
    def __init__(self) -> None:
        self.tp_score: dict[tuple[Position, int], Entry] = {}
        self.tp_move: dict[Position, Move] = {}
        self.history: set[Position] = set()
        self.root: Position | None = None
        # Positions on the line currently being searched, for repetition detection.
        self.path: set[Position] = set()
        self.nodes = 0
        self.deadline: float = 1 << 63
        self.soft: float = 1 << 63

    def bound(self, pos: Position, gamma: int, depth: int, root: bool = False) -> int:
        """Path-aware entry point: detect repetition along the line being searched.

        The reference only knows about repetition against positions already PLAYED
        (self.history). It has no idea when a line it is currently searching walks
        back into a position it already visited on that same line, so a forced
        perpetual check looks like a normal position and gets a normal score.

        Measured cost of that blindness: in four positions a reference engine
        scores as dead level BECAUSE of perpetual check, this engine returned +224,
        +240, +186 and -12 at depth 13-15, a mean absolute error of 166cp. That is
        why it walked a won game (+657) into a draw - it could not see that the
        checks never end - and why neither more time nor king-table tuning helped.

        Seeing a position twice on one line is treated as a draw, which is standard:
        a side able to repeat once can generally repeat again.

        Note this makes a node's value path-dependent, which the transposition table
        keyed on (pos, depth) cannot express - the classic graph-history problem. It
        is bounded here: the draw returns before anything is stored, and tp_score is
        cleared every search, so nothing survives beyond the current one.
        """
        # depth may arrive negative from reductions; treat that as quiescence.
        if depth > 0:
            if not root and pos in self.path:
                return 0
            self.path.add(pos)
            try:
                return self._node(pos, gamma, depth, root)
            finally:
                self.path.discard(pos)
        # Quiescence searches captures and promotions, which are irreversible, so a
        # line inside it cannot repeat and there is nothing worth tracking.
        return self._node(pos, gamma, depth, root)

    def _node(self, pos: Position, gamma: int, depth: int, root: bool = False) -> int:
        self.nodes += 1
        if self.nodes % 2048 == 0 and time.time() > self.deadline:
            raise Stop
        depth = max(depth, 0)
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER
        if not root:
            entry = self.tp_score.get((pos, depth), Entry(-MATE_UPPER, MATE_UPPER))
            if entry.lower >= gamma:
                return entry.lower
            if entry.upper < gamma:
                return entry.upper
            if depth > 0 and pos in self.history:
                return 0
        killer = self.tp_move.get(pos)

        def ceiling(v: int) -> int:
            if depth > 4 or v >= MATE_LOWER:
                return MATE_UPPER
            return pos.score + v + max(depth - 1, 0) * QS_A

        def moves() -> Iterator[tuple[int | None, Move | None]]:
            if 2 < depth < 6 and guard:
                yield None, None
            if depth == 0:
                yield None, None
            if killer and ((val := pos.value(killer)) >= QS or depth) and ceiling(val) >= gamma:
                yield val, killer
            # Staged generation. The reference values and sorts EVERY legal move at
            # every node; profiling showed the sort's cumulative cost was 5.39s of
            # 7.20s total, nearly all of it the gen_moves and value calls feeding
            # it. Most nodes cut off on an early capture, so splitting the list lets
            # that cutoff happen before the ~30 quiet moves are ever valued or
            # sorted. Measured +9.5% nodes.
            #
            # Ordering WITHIN each stage stays by value, descending, which is what
            # the consumer's futility break depends on: `if cap < gamma: break` is
            # only sound on a value-sorted stream. Splitting into two sorted streams
            # keeps that property inside each stage, and the break simply ends the
            # stage it is in rather than the whole list.
            board = pos.board
            captures: list[Move] = []
            quiets: list[Move] = []
            for m in pos.gen_moves():
                if board[m.j] in "pnbrqk" or m.prom:
                    captures.append(m)
                else:
                    quiets.append(m)

            scored = [(v, m) for m in captures if (v := pos.value(m)) >= QS or depth]
            scored.sort(reverse=True)
            yield from scored

            scored = [(v, m) for m in quiets if (v := pos.value(m)) >= QS or depth]
            scored.sort(reverse=True)
            yield from scored

        calm = abs(pos.score) < 750 and any(c in pos.board for c in "RBNQ")
        guard = not root and calm
        t = pos.score + NULL_MARGIN
        nmr = calm and depth >= 6 and -self.bound(pos.rotate(nullmove=True), 1 - t, depth - 7) >= t
        best, live = -MATE_UPPER, False
        for val, move in moves():
            if move is None and depth == 0:
                # Stand pat: the only static evaluation, so the only place the net
                # replaces the tables. Everything else keeps the incremental score.
                #
                # A residual net returns a CORRECTION to pos.score, clamped. The
                # clamp is what keeps the search coherent: sunfish prunes with
                # pos.score (futility ceilings, the null-move margin, the calmness
                # test), so an evaluation allowed to wander arbitrarily far from it
                # invalidates those margins. Measured, an unanchored net differed
                # from the tables by 311 cp on average and scored 13.8%.
                if _NNUE is None:
                    score = pos.score
                elif _NNUE.residual:
                    delta = _NNUE.evaluate(pos.board)
                    if delta > NNUE_MAX_CORRECTION:
                        delta = NNUE_MAX_CORRECTION
                    elif delta < -NNUE_MAX_CORRECTION:
                        delta = -NNUE_MAX_CORRECTION
                    score = pos.score + delta
                else:
                    score = _NNUE.evaluate(pos.board)
            elif move is None:
                if (cap := pos.score + EVAL_ROUGHNESS) >= gamma:
                    score = min(cap, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 4))
                    if score >= gamma and (proof := pos.king_capture()):
                        move, score, live = proof, MATE_UPPER, True
                else:
                    score = cap
            elif val is not None and val >= MATE_LOWER:
                score, live = MATE_UPPER, True
            else:
                assert val is not None
                if (cap := ceiling(val)) < gamma:
                    best = max(best, cap)
                    break
                move_depth = depth - 1 - (guard and depth >= 7 and val < LMR) - int(nmr)
                score = min(cap, -self.bound(pos.move(move), 1 - gamma, move_depth))
                live |= score > -MATE_UPPER
            best = max(best, score)
            if best >= gamma:
                if move is not None and depth:
                    self.tp_move[pos] = move
                    if len(self.tp_move) > TABLE_SIZE:
                        del self.tp_move[next(k for k in self.tp_move if k != self.root)]
                break

        if depth and not live and all(pos.move(m).king_capture() for m in pos.gen_moves()):
            mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
            best = mate if pos.rotate(nullmove=True).king_capture() else 0

        if not root:
            self.tp_score[pos, depth] = (
                Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
            )
        if len(self.tp_score) > TABLE_SIZE:
            del self.tp_score[next(iter(self.tp_score))]
        return best

    def search(
        self, history: list[Position]
    ) -> Iterator[tuple[int, int, int, "Move | None"]]:
        """Iterative deepening MTD-bi search."""
        self.nodes, self.history, self.tp_score = 0, set(history), {}
        self.path.clear()
        pos = self.root = history[-1]
        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END
        gamma = 0
        for depth in range(1, 1000):
            lower, upper = 1 - MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth, root=True)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(pos)
                gamma = (lower + upper + 1) // 2
            if time.time() > self.soft:
                return


###############################################################################
# python-chess <-> sunfish bridge
###############################################################################


def square_to_sunfish(s: int) -> int:
    return A1 + chess.square_file(s) - 10 * chess.square_rank(s)


def sunfish_to_square(i: int) -> int:
    # Mirror sunfish's render(): file = (i - A1) % 10, rank_1to8 = 1 - (i - A1) // 10.
    f = (i - A1) % 10
    r = -((i - A1) // 10)
    return chess.square(f, r)


def board_to_sunfish(b: chess.Board) -> Position:
    """Build a sunfish Position from a python-chess board, always oriented so
    the side to move is 'white' (uppercase) in sunfish's frame."""
    cells = [" "] * 120
    for r in range(8):
        for f in range(8):
            s = chess.square(f, r)
            p = b.piece_at(s)
            cells[square_to_sunfish(s)] = p.symbol() if p else "."
    # The pad columns/rows are spaces, which are in sunfish's edge-break set
    # (" \nPNBRQK"), so a plain join is a valid off-board sentinel.
    board_str = "".join(cells)
    score = 0
    for i, ch in enumerate(board_str):
        if ch in "PNBRQK":
            score += pst[ch][i]
        elif ch in "pnbrqk":
            score -= pst[ch.upper()][119 - i]
    wc = (b.has_queenside_castling_rights(chess.WHITE), b.has_kingside_castling_rights(chess.WHITE))
    bc = (b.has_queenside_castling_rights(chess.BLACK), b.has_kingside_castling_rights(chess.BLACK))
    ep = square_to_sunfish(b.ep_square) if b.ep_square is not None else 0
    pos = Position(board_str, score, wc, bc, ep, 0)
    return pos if b.turn == chess.WHITE else pos.rotate()


def sunfish_move_to_uci(m: Move, b: chess.Board) -> str:
    i, j, prom = m
    if b.turn == chess.BLACK:
        i, j = 119 - i, 119 - j
    uci = chess.square_name(sunfish_to_square(i)) + chess.square_name(sunfish_to_square(j))
    return uci + (prom.lower() if prom else "")


###############################################################################
# Competition entrypoint
###############################################################################

# The process stays alive across moves in a game, so reuse one searcher: its
# transposition tables and killer moves carry over between our own moves.
_searcher = Searcher()

# Every position we have been asked about this game, oldest first. The searcher
# scores a repeated position as a draw, so this is what stops us shuffling a won
# game into a threefold the referee then claims. Module state is per game, so it
# starts empty for each new game and never leaks into the next one.
_history: list[Position] = []

# Milliseconds we shave off the clock for reply/accounting lag (AGENTS.md).
DELAY_MS = 200


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in ``fen``.

    Sunfish searches under a wall-clock budget derived from our remaining
    clock. The time control is 120s + 0.5s/move per side, so a fortieth of the
    clock plus the per-move increment, minus reply lag, keeps us well clear of
    a flag while still thinking hard in the middlegame.
    """
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return ""
    if len(legal) == 1:
        # Nothing to think about, and it banks the increment.
        return legal[0].uci()

    pos = board_to_sunfish(board)
    _history.append(pos)

    # THREE NUMBERS, MILLISECONDS. `budget` is what this move is worth: a
    # fortieth of the clock plus the increment it earns back, less reply lag.
    # `soft` is when to stop STARTING a new iteration, `think` the wall one
    # iteration may run to. Letting think exceed soft is what buys depth in
    # sharp positions; both are clamped well under the clock so a slow
    # iteration still cannot flag us.
    # Budget the clock we were handed across the moves we expect to still play,
    # rather than from a constant. The reference formula (remaining/40 + 500, with
    # a 5x overrun allowance) front-loads catastrophically: simulated over a game
    # it spends 13.3s on move one, has burnt 118 of its 120 seconds by move 20,
    # and then plays every remaining move on 0.3-0.6s with a 1.4s buffer. That is
    # both where the measured blunders happen and an uncomfortable flag risk.
    #
    # Dividing by the moves left instead gives ~2.5s a move through the whole
    # middlegame - about 5x the old budget at move 40 - and keeps ~8.5s spare.
    #
    # No increment constant appears here on purpose. Beyond being the wrong thing
    # to hardcode, an absolute constant makes the formula scale-dependent, so it
    # could only ever be tested at the real 120s control, seven minutes a game. As
    # a pure fraction of the clock, a fast game is a faithful scale model of a slow
    # one and the formula can be measured over hundreds of games.
    remaining = max(time_left_ms - DELAY_MS, 0)
    moves_left = max(20, 60 - board.fullmove_number)
    budget_ms = remaining / moves_left
    soft_ms = max(min(budget_ms, remaining / 4), 20)
    think_ms = max(min(1.5 * budget_ms, remaining / 3), 40)

    start = time.time()
    _searcher.soft = start + soft_ms / 1000
    _searcher.deadline = start + think_ms / 1000

    # Only a COMPLETED depth's last fail-high is trustworthy: a stop inside a
    # depth can catch a probe at a nonsense window, so we keep the finished
    # depth's move (`best`) and only promote the in-progress one (`cand`) once
    # the next depth begins.
    best: Move | None = None
    cand: Move | None = None
    seen_depth = 1
    try:
        for depth, gamma, score, move in _searcher.search(_history):
            if depth > seen_depth:
                best, seen_depth = cand or best, depth
            if score >= gamma and move is not None:
                cand = move
    except Stop:
        cand = best or cand

    chosen = cand or best
    if chosen is not None:
        uci = sunfish_move_to_uci(chosen, board)
        try:
            if chess.Move.from_uci(uci) in board.legal_moves:
                return uci
        except ValueError:
            pass

    # Fallback: never forfeit on a malformed/illegal engine move.
    return legal[0].uci()


# Warm the search once at import so the first real move does not pay first-run
# cost on the clock (import time has its own 60s budget).
def _warmup() -> None:
    warm = Searcher()
    warm.deadline = time.time() + 0.2
    warm.soft = warm.deadline
    start = board_to_sunfish(chess.Board())
    try:
        for _ in warm.search([start]):
            pass
    except Stop:
        pass


_warmup()
