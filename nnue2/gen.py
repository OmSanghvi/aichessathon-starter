"""Generate training data: diverse positions labelled by a reference engine.

Stores the RAW BOARD, not feature indices, so the feature set can be redesigned
without throwing away hours of labelling. That was a real cost in v1.

The labelling engine runs here, on your machine, only to annotate positions. It is
never packaged into the submission - the rules allow learning from engine-annotated
data and forbid shipping an engine.

Resumable: existing shards are counted, so stopping and restarting is free.

    .venv/bin/python nnue2/gen.py --hours 8
"""

import argparse
import contextlib
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nnue2.features import EVAL_CLAMP, board_to_codes, pack_boards  # noqa: E402

DEFAULT_ENGINE = "/opt/homebrew/bin/stockfish"

# Self-play move budget. Tiny on purpose: we want varied, broadly sensible
# positions, and every millisecond here is one not spent labelling. In v1 driving
# self-play through the full get_move budget made generation 18x slower.
PLAY_THINK_S = 0.01

# Short, legal opening lines which reach normal castled middlegames. Random
# playouts from the initial position almost never castle, which left the original
# data with less than 3% coverage of the king-side edge buckets. The competition
# begins from curated, near-level positions, so teaching king safety from these
# ordinary structures is much more valuable than collecting another million
# uncastled random walks.
CASTLED_OPENINGS = (
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1 f8c5 d2d3 e8g8",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5 c4d5 f6d5 f1g2 b8c6 g1f3 f8c5 e1g1",
    "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6 b1c3 f8e7 c1g5 e8g8 e2e3",
)


class Labeller:
    def __init__(self, path: str, depth: int) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        # One thread: deterministic, and thread overhead dominates at these depths.
        with contextlib.suppress(Exception):
            self.engine.configure({"Threads": 1, "Hash": 128})
        self.limit = chess.engine.Limit(depth=depth)

    def score(self, board: chess.Board) -> int | None:
        try:
            info = self.engine.analyse(board, self.limit)
        except chess.engine.EngineError:
            return None
        cp = info["score"].pov(board.turn).score(mate_score=EVAL_CLAMP)
        if cp is None:
            return None
        return int(max(-EVAL_CLAMP, min(EVAL_CLAMP, cp)))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.engine.quit()


def random_opening(rng: random.Random, max_plies: int) -> chess.Board:
    """A varied but not absurd opening.

    Rated games start from curated positions rather than the standard start, so
    opening diversity matters more here than for a normal engine: a net that only
    ever saw positions downstream of 1.e4 would be lost in exactly the games that
    count.
    """
    board = chess.Board()
    for _ in range(rng.randint(2, max_plies)):
        moves = list(board.legal_moves)
        if not moves:
            break
        weights = []
        for m in moves:
            w = 1.0
            if board.is_capture(m):
                w += 1.5
            f, r = chess.square_file(m.to_square), chess.square_rank(m.to_square)
            if 2 <= f <= 5 and 2 <= r <= 5:
                w += 0.8
            weights.append(w)
        board.push(rng.choices(moves, weights=weights, k=1)[0])
        if board.is_game_over():
            break
    return board


def castled_opening(rng: random.Random) -> chess.Board:
    """Return one verified normal opening position with at least one castling."""
    board = chess.Board()
    for uci in rng.choice(CASTLED_OPENINGS).split():
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"invalid built-in opening move {uci}")
        board.push(move)
    return board


def opening_position(
    rng: random.Random, max_plies: int, castle_opening_share: float,
) -> chess.Board:
    """Mix broad random openings with structures that teach king safety."""
    if rng.random() < castle_opening_share:
        return castled_opening(rng)
    return random_opening(rng, max_plies)


def play_game(engine: object, board: chess.Board, rng: random.Random,
              max_plies: int, deviate: float) -> list[chess.Board]:
    """Play a cheap current-Nengine game, preserving broad position diversity.

    This deliberately calls the mailbox engine's Python bridge rather than the
    former Sunfish-only API.  It is offline data generation, never submission
    code, and the reference labels below remain separate from move selection.
    """
    import agent

    # A submitted process handles exactly one game.  Reproduce that boundary for
    # repetition history while retaining the TT and measured NPS between games.
    engine.game_hashes[:] = 0
    engine.game_count = 0
    seen: list[chess.Board] = []
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        seen.append(board.copy())
        legal = list(board.legal_moves)
        if not legal:
            break
        move = legal[0]
        try:
            core = engine.search(board, PLAY_THINK_S, max(0.02, 2 * PLAY_THINK_S))
            if core != 0:
                parsed = chess.Move.from_uci(agent._move_to_uci(core))
                if parsed in board.legal_moves:
                    move = parsed
        except Exception:
            # Data diversity matters more than a perfect self-play move.  A
            # legal fallback keeps an intermittent search failure from losing a
            # multi-hour resumable run.
            move = legal[0]
        if rng.random() < deviate:
            move = rng.choice(legal)
        board.push(move)
    return seen


def main() -> None:
    p = argparse.ArgumentParser(description="Generate NNUE training data.")
    p.add_argument("--hours", type=float, default=8.0, help="wall-clock budget")
    p.add_argument(
        "--data-dir", default=str(Path(__file__).resolve().parent / "data"),
        help="resumable shard directory (defaults to nnue2/data)",
    )
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--depth", type=int, default=10,
                   help="labelling depth; 10 is close to 14 and much faster")
    p.add_argument("--engine-path", default=DEFAULT_ENGINE)
    p.add_argument("--max-plies", type=int, default=90)
    p.add_argument("--opening-plies", type=int, default=16)
    p.add_argument(
        "--castle-opening-share", type=float, default=0.50,
        help="fraction of games seeded from ordinary castled openings",
    )
    p.add_argument("--deviate", type=float, default=0.06)
    # 10 rather than 6: at 6 the sample averaged 15.9 pieces a position, skewed
    # toward bare endgames that search already handles well. Raising the floor
    # shifts the set toward the middlegames where evaluation decides moves.
    p.add_argument("--min-pieces", type=int, default=10)
    p.add_argument("--keep-above", type=int, default=1500,
                   help="treat |cp| above this as already decided")
    p.add_argument("--keep-extreme", type=float, default=0.15,
                   help="fraction of decided positions to keep")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    if not 0.0 <= args.castle_opening_share <= 1.0:
        raise SystemExit("--castle-opening-share must be between 0 and 1")

    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    labeller = Labeller(args.engine_path, args.depth)

    shard = len(list(data.glob("shard_*.npz")))
    existing = 0
    for path in data.glob("shard_*.npz"):
        with np.load(path) as z:
            existing += z["labels"].shape[0]
    print(f"resuming: {shard} shard(s), {existing:,} positions already on disk",
          flush=True)

    boards: list[np.ndarray] = []
    turns: list[int] = []
    labels: list[int] = []
    written = 0
    start = time.time()
    deadline = start + args.hours * 3600

    def flush() -> None:
        nonlocal shard, boards, turns, labels, written
        if not boards:
            return
        path = data / f"shard_{shard:04d}.npz"
        np.savez_compressed(
            path,
            boards=pack_boards(boards),
            turns=np.array(turns, dtype=np.int8),
            labels=np.array(labels, dtype=np.int16),
        )
        written += len(boards)
        rate = written / max(time.time() - start, 1e-9)
        remaining = max(deadline - time.time(), 0)
        print(f"  wrote {path.name}: {len(boards):,} rows  "
              f"({written:,} this run, {existing + written:,} total)  "
              f"{rate:.0f}/sec  {remaining / 3600:.1f}h left", flush=True)
        boards, turns, labels = [], [], []
        shard += 1

    try:
        import agent

        self_player = agent._Engine()
        while time.time() < deadline:
            opening = opening_position(
                rng, args.opening_plies, args.castle_opening_share
            )
            for board in play_game(
                self_player, opening, rng, args.max_plies, args.deviate
            ):
                if time.time() >= deadline:
                    break
                if board.is_game_over() or len(board.piece_map()) < args.min_pieces:
                    continue
                cp = labeller.score(board)
                if cp is None:
                    continue
                # Already-decided positions teach little and would dominate the loss.
                if abs(cp) >= args.keep_above and rng.random() > args.keep_extreme:
                    continue
                codes, turn = board_to_codes(board)
                boards.append(codes)
                turns.append(turn)
                labels.append(cp)
            if len(boards) >= args.shard_size:
                flush()
    except KeyboardInterrupt:
        print("\ninterrupted; flushing what we have", flush=True)
    finally:
        flush()
        labeller.close()

    total = existing + written
    elapsed = time.time() - start
    print(f"\ndone: {written:,} new positions in {elapsed / 3600:.2f}h "
          f"({written / max(elapsed, 1e-9):.0f}/sec)")
    print(f"total on disk: {total:,}")
    if total < 2_000_000:
        print("NOTE: v1 plateaued at 500k and lost badly. Under ~2M expect the same.")


if __name__ == "__main__":
    main()
