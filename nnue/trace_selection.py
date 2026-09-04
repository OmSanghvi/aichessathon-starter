"""Trace what the move-selection loop sees at a budget that misbehaves.

The engine returns the right move at 2s, 15s, 60s and 240s but a losing one at
5s, so the search understands the position and the SELECTION is what breaks. This
replays get_move's loop with logging to show which (depth, gamma, score, move)
tuples arrive and which one we end up committing to.
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

FEN = "6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42"


def trace(clock_ms: int) -> None:
    board = chess.Board(FEN)
    pos = agent.board_to_sunfish(board)

    remaining = max(clock_ms - agent.DELAY_MS, 0)
    budget_ms = remaining / 40 + 500
    soft_ms = max(min(budget_ms, remaining / 4), 20)
    think_ms = max(min(5 * budget_ms, remaining / 2), 40)

    searcher = agent.Searcher()
    start = time.time()
    searcher.soft = start + soft_ms / 1000
    searcher.deadline = start + think_ms / 1000

    print(f"\n=== clock {clock_ms} ms  soft={soft_ms:.0f}ms  think={think_ms:.0f}ms ===")
    best = None
    cand = None
    seen_depth = 1
    rows = 0
    stopped = False
    try:
        for depth, gamma, score, move in searcher.search([pos]):
            promoted = ""
            if depth > seen_depth:
                best, seen_depth = cand or best, depth
                promoted = f"  [new depth {depth}: best <- {_san(board, best)}]"
            accepted = ""
            if score >= gamma and move is not None:
                cand = move
                accepted = "  cand <- " + _san(board, move)
            rows += 1
            if rows <= 60:
                print(
                    f"  d={depth:<2} gamma={gamma:>7} score={score:>7} "
                    f"move={_san(board, move):<8}{accepted}{promoted}"
                )
    except agent.Stop:
        stopped = True
        cand = best or cand

    chosen = cand or best
    print(f"  ... {rows} yields, stop={stopped}")
    print(f"  FINAL: cand={_san(board, cand)}  best={_san(board, best)} "
          f"-> chosen={_san(board, chosen)}")


def _san(board: chess.Board, move: "agent.Move | None") -> str:
    if move is None:
        return "-"
    uci = agent.sunfish_move_to_uci(move, board)
    try:
        m = chess.Move.from_uci(uci)
        return board.san(m) if m in board.legal_moves else uci
    except ValueError:
        return uci


def main() -> None:
    for clock in (2_000, 5_000, 15_000):
        trace(clock)


if __name__ == "__main__":
    main()
