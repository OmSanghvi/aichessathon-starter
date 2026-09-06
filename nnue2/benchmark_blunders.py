"""Regression benchmark for high-swing mistakes made by the live agent.

This is an OFFLINE development tool.  It imports the root ``agent.py`` under
test and uses a local UCI reference solely to measure move loss.  It is never
imported by the submission.

Run it before and after a general search/evaluation change:

    uv run python nnue2/benchmark_blunders.py --think-ms 2200 --depth 14
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nnue2.features import EVAL_CLAMP  # noqa: E402

DEFAULT_ENGINE = "/opt/homebrew/bin/stockfish"


def score_cp(score: chess.engine.PovScore, color: chess.Color) -> int:
    """Return a bounded score from ``color``'s perspective."""
    value = score.pov(color).score(mate_score=EVAL_CLAMP)
    if value is None:
        return 0
    return int(max(-EVAL_CLAMP, min(EVAL_CLAMP, value)))


def agent_move(agent_module: object, board: chess.Board, think_ms: int) -> chess.Move:
    """Call the submitted mailbox search with a conservative legal fallback."""
    legal = list(board.legal_moves)
    if len(legal) == 1:
        return legal[0]
    player = agent_module._Engine()
    try:
        packed = player.search(board, think_ms / 1000.0, 2.0 * think_ms / 1000.0)
        move = chess.Move.from_uci(agent_module._move_to_uci(packed))
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    return legal[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the current agent across its mined blunder suite."
    )
    parser.add_argument("--data", default="nnue2/blunders/mistakes.jsonl")
    parser.add_argument("--think-ms", type=int, default=2200)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--engine-path", default=DEFAULT_ENGINE)
    parser.add_argument("--json-out", type=Path,
                        help="optional detailed result file for candidate comparison")
    args = parser.parse_args()
    if args.think_ms < 1 or args.depth < 1:
        raise SystemExit("--think-ms and --depth must be positive")

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    if not rows:
        raise SystemExit(f"no positions in {args.data}")

    import agent

    reference = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    with contextlib.suppress(chess.engine.EngineError):
        reference.configure({"Threads": 1, "Hash": 128})
    limit = chess.engine.Limit(depth=args.depth)
    results: list[dict[str, int | str]] = []
    try:
        for number, row in enumerate(rows, 1):
            board = chess.Board(row["fen"])
            root_color = board.turn
            best_info = reference.analyse(board, limit)
            pv = best_info.get("pv", [])
            if not pv:
                continue
            best = pv[0]
            best_cp = score_cp(best_info["score"], root_color)
            played = agent_move(agent, board, args.think_ms)
            child = board.copy(stack=False)
            child.push(played)
            reply_info = reference.analyse(child, limit)
            played_cp = -score_cp(reply_info["score"], child.turn)
            loss = max(0, best_cp - played_cp)
            result: dict[str, int | str] = {
                "fen": board.fen(), "best": best.uci(), "played": played.uci(),
                "best_cp": best_cp, "played_cp": played_cp, "loss": loss,
            }
            results.append(result)
            print(f"{number:2}/{len(rows)} loss {loss:4}cp  "
                  f"{played.uci()} (best {best.uci()})", flush=True)
    finally:
        reference.quit()

    losses = sorted(int(item["loss"]) for item in results)
    if not losses:
        raise SystemExit("reference returned no principal variations")
    mean = sum(losses) / len(losses)
    median = losses[len(losses) // 2]
    severe = sum(loss >= 120 for loss in losses)
    exact = sum(item["played"] == item["best"] for item in results)
    print(f"\npositions {len(losses)} | mean loss {mean:.1f}cp | "
          f"median {median}cp | >=120cp {severe} | best move {exact}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "think_ms": args.think_ms, "depth": args.depth,
            "mean_loss_cp": mean, "median_loss_cp": median,
            "severe": severe, "exact": exact, "positions": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
