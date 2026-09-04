"""Generate NNUE training data: diverse positions plus evaluation labels.

Positions come from self-play with randomised openings. Rated games start from
curated positions rather than the standard start, so opening diversity matters
more here than it would for a normal engine: a net that only ever saw positions
downstream of 1.e4 would be out of its depth in exactly the games that count.

Labels come from a deep search by our own engine (`--labeller self`), which is
distillation: the net learns to predict what a deep search concludes from a
static position, which is a real step up from the static table it replaces.
`--labeller uci` uses an external engine binary instead, for stronger labels.
That binary is only ever used here, on your machine, to annotate data; it never
ships in the submission, which is what the rules allow.

Output: resumable .npz shards in nnue/data/, each with
    features  int32 [rows, 32]  active feature indices, -1 padded
    labels    int16 [rows]      centipawns, side-to-move relative, clamped

Run (background friendly):
    .venv/bin/python nnue/gen_data.py --games 200 --shard-size 20000
"""

import argparse
import contextlib
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nnue.features import EVAL_CLAMP, pack_batch  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

# Self-play move budget. Deliberately tiny: we want varied, broadly-sensible
# positions, and every millisecond here is one not spent on labelling.
PLAY_THINK_S = 0.01


# ---------------------------------------------------------------------------
# Labellers
# ---------------------------------------------------------------------------


class SelfLabeller:
    """Label a position with a deep search from our own engine."""

    def __init__(self, think_s: float = 0.10) -> None:
        import agent

        self.agent = agent
        self.think_s = think_s

    def score(self, board: chess.Board) -> int | None:
        agent = self.agent
        pos = agent.board_to_sunfish(board)
        searcher = agent.Searcher()
        now = time.time()
        searcher.soft = searcher.deadline = now + self.think_s
        best_score: int | None = None
        try:
            for _depth, gamma, score, _move in searcher.search([pos]):
                # A fail-high is a trustworthy lower bound on the value.
                if score >= gamma:
                    best_score = score
        except agent.Stop:
            pass
        if best_score is None:
            return None
        # Sunfish scores are already side-to-move relative centipawns.
        return int(max(-EVAL_CLAMP, min(EVAL_CLAMP, best_score)))


class UciLabeller:
    """Label with an external UCI engine.

    TRAINING DATA ONLY. This binary lives outside the repo and is never packaged
    into the submission; the rules allow learning from engine-annotated data but
    forbid shipping an engine, so nothing here may ever reach the zip.
    """

    def __init__(self, path: str, depth: int = 10) -> None:
        import chess.engine

        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        # One thread keeps labels deterministic and avoids thread overhead that
        # dominates at the shallow depths we label with.
        with contextlib.suppress(Exception):
            self.engine.configure({"Threads": 1, "Hash": 64})
        self.depth = depth

    def score(self, board: chess.Board) -> int | None:
        import chess.engine

        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        pov = info["score"].pov(board.turn)
        cp = pov.score(mate_score=EVAL_CLAMP)
        if cp is None:
            return None
        return int(max(-EVAL_CLAMP, min(EVAL_CLAMP, cp)))

    def close(self) -> None:
        self.engine.quit()


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------


def random_opening(rng: random.Random, plies: int) -> chess.Board:
    """A randomised but not absurd opening, for position diversity."""
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        # Prefer captures/central moves a little so we do not only get junk.
        weights = []
        for m in moves:
            w = 1.0
            if board.is_capture(m):
                w += 1.5
            to_file, to_rank = chess.square_file(m.to_square), chess.square_rank(m.to_square)
            if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
                w += 0.8
            weights.append(w)
        board.push(rng.choices(moves, weights=weights, k=1)[0])
        if board.is_game_over():
            break
    return board


def self_play_positions(
    board: chess.Board,
    labeller: SelfLabeller | UciLabeller,
    rng: random.Random,
    max_plies: int,
    deviate: float = 0.15,
) -> list[chess.Board]:
    """Play a game from ``board`` with our engine, returning positions seen."""
    import agent

    seen: list[chess.Board] = []
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        seen.append(board.copy())
        legal = list(board.legal_moves)
        if not legal:
            break
        # Position QUALITY does not matter much here, only diversity and being
        # broadly sensible, so drive self-play with a very short search rather
        # than get_move's real time budget. Going through get_move spent ~600ms
        # a move and made generation ~20x slower than the labelling it feeds.
        move = legal[0]
        pos = agent.board_to_sunfish(board)
        searcher = agent.Searcher()
        searcher.soft = searcher.deadline = time.time() + PLAY_THINK_S
        try:
            for _d, gamma, score, m in searcher.search([pos]):
                if m is not None and score >= gamma:
                    move = m
                    break
        except agent.Stop:
            pass
        if isinstance(move, agent.Move):
            uci = agent.sunfish_move_to_uci(move, board)
            try:
                parsed = chess.Move.from_uci(uci)
            except ValueError:
                parsed = legal[0]
            move = parsed if parsed in board.legal_moves else legal[0]
        # Occasionally deviate, so we do not record one deterministic line.
        if rng.random() < deviate:
            move = rng.choice(legal)
        board.push(move)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NNUE training data.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--think-s", type=float, default=0.10)
    parser.add_argument("--labeller", choices=("self", "uci"), default="self")
    parser.add_argument("--engine-path", default="")
    parser.add_argument("--uci-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deviate", type=float, default=0.08, help="chance of a random move in self-play"
    )
    parser.add_argument(
        "--keep-above", type=int, default=1500, help="treat |cp| above this as already decided"
    )
    parser.add_argument(
        "--keep-extreme", type=float, default=0.15, help="fraction of decided positions to keep"
    )
    parser.add_argument(
        "--min-pieces", type=int, default=8, help="skip positions barer than this"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    labeller: SelfLabeller | UciLabeller
    if args.labeller == "uci":
        if not args.engine_path:
            raise SystemExit("--labeller uci needs --engine-path")
        labeller = UciLabeller(args.engine_path, args.uci_depth)
    else:
        labeller = SelfLabeller(args.think_s)

    from nnue.features import features_from_board

    shard = len(list(DATA_DIR.glob("shard_*.npz")))
    feats: list[np.ndarray] = []
    labels: list[int] = []
    written = 0
    start = time.time()

    for game in range(args.games):
        # Vary how deep the random opening runs so positions spread across game
        # phases instead of clustering wherever a fixed opening length lands.
        opening_plies = rng.randint(2, max(2, args.opening_plies))
        opening = random_opening(rng, opening_plies)
        positions = self_play_positions(
            opening, labeller, rng, args.max_plies, args.deviate
        )
        for board in positions:
            if board.is_game_over():
                continue
            # Bare endings teach the net little that search does not already
            # handle, and they were skewing the set toward 14 pieces a position.
            if len(board.piece_map()) < args.min_pieces:
                continue
            cp = labeller.score(board)
            if cp is None:
                continue
            # Already-decided positions teach the net little and would dominate
            # the loss, so keep only a fraction of the blown-out ones.
            if abs(cp) >= args.keep_above and rng.random() > args.keep_extreme:
                continue
            feats.append(features_from_board(board))
            labels.append(cp)

        if len(feats) >= args.shard_size:
            path = DATA_DIR / f"shard_{shard:04d}.npz"
            np.savez_compressed(
                path, features=pack_batch(feats), labels=np.array(labels, dtype=np.int16)
            )
            written += len(feats)
            print(
                f"game {game + 1}/{args.games}: wrote {path.name} "
                f"({len(feats):,} rows, {written:,} total, {time.time() - start:.0f}s)",
                flush=True,
            )
            feats, labels, shard = [], [], shard + 1

    if feats:
        path = DATA_DIR / f"shard_{shard:04d}.npz"
        np.savez_compressed(
            path, features=pack_batch(feats), labels=np.array(labels, dtype=np.int16)
        )
        written += len(feats)
        print(f"wrote {path.name} ({len(feats):,} rows)", flush=True)

    if isinstance(labeller, UciLabeller):
        labeller.close()
    elapsed = time.time() - start
    rate = written / elapsed if elapsed > 0 else 0.0
    print(f"done: {written:,} labelled positions in {elapsed:.0f}s ({rate:.1f}/sec)", flush=True)
    if rate > 0:
        for target in (100_000, 1_000_000):
            print(f"  at this rate {target:,} positions takes {target / rate / 3600:.1f} hours")


if __name__ == "__main__":
    main()
