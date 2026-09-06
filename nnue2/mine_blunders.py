"""Mine high-swing current-engine mistakes into trainable hard examples.

This is an OFFLINE tool.  It compares Nengine's move at a chosen thinking budget
against a local UCI reference, writes the best and played child positions in the
same raw-board shard format as ``nnue2/gen.py``, and records a JSONL audit trail.
Nothing here is imported by ``agent.py`` or included in ``submission.zip``.

Example (resume-safe):

    uv run python nnue2/mine_blunders.py --games 100 --think-ms 2200 --depth 14
    uv run python nnue2/train.py --hard-data-dir nnue2/blunders --hard-repeat 4 ...
"""

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nnue2.features import EVAL_CLAMP, board_to_codes, pack_boards  # noqa: E402
from nnue2.gen import opening_position  # noqa: E402

DEFAULT_ENGINE = "/opt/homebrew/bin/stockfish"


def cp(score: chess.engine.PovScore, color: chess.Color) -> int:
    """Bound a reference score in the same convention used by training labels."""
    value = score.pov(color).score(mate_score=EVAL_CLAMP)
    return 0 if value is None else int(max(-EVAL_CLAMP, min(EVAL_CLAMP, value)))


def choose(agent_module: object, player: object, board: chess.Board, think_ms: int) -> chess.Move:
    """Choose exactly as the current mailbox engine does, with a safe fallback."""
    legal = list(board.legal_moves)
    if len(legal) == 1:
        return legal[0]
    try:
        core = player.search(board, think_ms / 1000.0, 2.0 * think_ms / 1000.0)
        if core != 0:
            move = chess.Move.from_uci(agent_module._move_to_uci(core))
            if move in board.legal_moves:
                return move
    except Exception:
        pass
    return legal[0]


def append_child(
    board: chess.Board, label: int, boards: list[np.ndarray],
    turns: list[int], labels: list[int],
) -> None:
    codes, turn = board_to_codes(board)
    boards.append(codes)
    turns.append(turn)
    labels.append(label)


def write_shard(
    out: Path, shard: int, boards: list[np.ndarray], turns: list[int],
    labels: list[int], mined: int,
) -> int:
    """Persist one completed-game batch, so an interrupted audit keeps its data."""
    if not boards:
        return shard
    path = out / f"shard_{shard:04d}.npz"
    np.savez_compressed(
        path,
        boards=pack_boards(boards),
        turns=np.array(turns, dtype=np.int8),
        labels=np.array(labels, dtype=np.int16),
    )
    print(f"wrote {path}: {len(boards):,} child positions through {mined} blunders",
          flush=True)
    boards.clear()
    turns.clear()
    labels.clear()
    return shard + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine current-engine blunders offline.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--opening-plies", type=int, default=16)
    parser.add_argument("--castle-opening-share", type=float, default=0.75)
    parser.add_argument("--think-ms", type=int, default=2200)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--threshold", type=int, default=120,
                        help="minimum centipawn loss to retain")
    parser.add_argument("--sample-every", type=int, default=2,
                        help="audit every Nth ply to control reference-engine cost")
    parser.add_argument("--engine-path", default=DEFAULT_ENGINE)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "blunders"))
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.games < 1 or args.max_plies < 1 or args.sample_every < 1:
        raise SystemExit("--games, --max-plies and --sample-every must be positive")
    if not 0.0 <= args.castle_opening_share <= 1.0:
        raise SystemExit("--castle-opening-share must be between 0 and 1")

    import agent

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = len(list(out.glob("shard_*.npz")))
    metadata = out / "mistakes.jsonl"
    rng = random.Random(args.seed)
    reference = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    with contextlib.suppress(chess.engine.EngineError):
        reference.configure({"Threads": 1, "Hash": 128})
    limit = chess.engine.Limit(depth=args.depth)

    boards: list[np.ndarray] = []
    turns: list[int] = []
    labels: list[int] = []
    mined = audited = 0
    try:
        with metadata.open("a", encoding="utf-8") as audit:
            for game in range(args.games):
                board = opening_position(
                    rng, args.opening_plies, args.castle_opening_share
                )
                player = agent._Engine()
                for ply in range(args.max_plies):
                    if board.is_game_over(claim_draw=True):
                        break
                    played = choose(agent, player, board, args.think_ms)
                    if ply % args.sample_every == 0:
                        audited += 1
                        root_color = board.turn
                        best_info = reference.analyse(board, limit)
                        best_pv = best_info.get("pv", [])
                        if best_pv:
                            best = best_pv[0]
                            best_cp = cp(best_info["score"], root_color)
                            child = board.copy(stack=False)
                            child.push(played)
                            played_info = reference.analyse(child, limit)
                            played_cp = -cp(played_info["score"], child.turn)
                            swing = best_cp - played_cp
                            if swing >= args.threshold:
                                best_child = board.copy(stack=False)
                                best_child.push(best)
                                # The label is always side-to-move relative.
                                append_child(best_child, -best_cp, boards, turns, labels)
                                append_child(child, -played_cp, boards, turns, labels)
                                audit.write(json.dumps({
                                    "fen": board.fen(), "best": best.uci(),
                                    "played": played.uci(), "best_cp": best_cp,
                                    "played_cp": played_cp, "swing": swing,
                                }) + "\n")
                                mined += 1
                    board.push(played)
                print(f"game {game + 1}/{args.games}: audited {audited}, "
                      f"mined {mined}", flush=True)
                # Complete games are independent.  Persist immediately instead
                # of losing an hours-long audit if the machine is interrupted.
                audit.flush()
                shard = write_shard(out, shard, boards, turns, labels, mined)
    finally:
        reference.quit()

    shard = write_shard(out, shard, boards, turns, labels, mined)
    if mined == 0:
        print("no mistakes met the threshold; no shard written")


if __name__ == "__main__":
    main()
