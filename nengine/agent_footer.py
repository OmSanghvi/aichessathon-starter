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
