"""Compare move-selection rules across many budgets, scored by a reference engine.

The engine finds the right move at most budgets and a losing one at others, which
points at HOW we pick a move out of the iterative-deepening stream rather than at
the search itself. Two candidate rules:

  current : chosen = cand or best   (prefer the newest fail-high, even mid-bracket)
  proposed: chosen = best or cand   (prefer the last COMPLETED depth's conclusion)

A mid-bracket fail-high at a low window only proves "value >= that window", which
is a weak lower bound and not evidence the move is best. This measures how often
each rule produces a move the reference engine considers a blunder.
"""

import sys
import time
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

ENGINE = "/opt/homebrew/bin/stockfish"
BLUNDER_CP = 150

# Positions where the engine has shown instability, plus ordinary middlegames.
POSITIONS = [
    "6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42",
    "6k1/4R3/2r1p1Bp/1p6/1n2K P2/4r1P1/1P1p3P/3R4 w - - 7 42".replace(" P2", "3P2"),
    "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7",
    "rnbqk2r/pp2nppp/4p3/2ppP3/3P4/P1P5/2P2PPP/1RBQKBNR b Kkq - 2 7",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "4rrk1/pp1n1ppp/2pb4/3p4/3P4/2NBP3/PP3PPP/2R2RK1 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]

BUDGETS = (2_000, 3_500, 5_000, 8_000, 12_000, 20_000, 40_000)


def pick(pos: "agent.Position", clock_ms: int, rule: str) -> "agent.Move | None":
    remaining = max(clock_ms - agent.DELAY_MS, 0)
    budget_ms = remaining / 40 + 500
    soft_ms = max(min(budget_ms, remaining / 4), 20)
    think_ms = max(min(5 * budget_ms, remaining / 2), 40)

    searcher = agent.Searcher()
    start = time.time()
    searcher.soft = start + soft_ms / 1000
    searcher.deadline = start + think_ms / 1000

    best = None
    cand = None
    seen_depth = 1
    try:
        for depth, gamma, score, move in searcher.search([pos]):
            if depth > seen_depth:
                best, seen_depth = cand or best, depth
            if score >= gamma and move is not None:
                cand = move
    except agent.Stop:
        cand = best or cand
    return (cand or best) if rule == "current" else (best or cand)


def main() -> None:
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "Hash": 128})
    limit = chess.engine.Limit(depth=16)

    totals = {"current": 0, "proposed": 0}
    losses = {"current": 0.0, "proposed": 0.0}
    tested = 0

    for fen in POSITIONS:
        board = chess.Board(fen)
        ref = engine.analyse(board, limit)
        base = ref["score"].pov(board.turn).score(mate_score=10000)
        print(f"\n{fen}\n  reference {board.san(ref['pv'][0])} at {base} cp")

        for clock in BUDGETS:
            row = f"    {clock:>6}ms:"
            for rule in ("current", "proposed"):
                mv = pick(agent.board_to_sunfish(board), clock, rule)
                if mv is None:
                    row += f"  {rule}=none"
                    continue
                uci = agent.sunfish_move_to_uci(mv, board)
                try:
                    parsed = chess.Move.from_uci(uci)
                except ValueError:
                    parsed = None
                if parsed is None or parsed not in board.legal_moves:
                    row += f"  {rule}=ILLEGAL"
                    continue
                board.push(parsed)
                after = engine.analyse(board, limit)["score"].pov(
                    not board.turn
                ).score(mate_score=10000)
                board.pop()
                drop = base - after
                mark = "!" if drop >= BLUNDER_CP else " "
                if drop >= BLUNDER_CP:
                    totals[rule] += 1
                losses[rule] += max(drop, 0)
                row += f"  {rule}={board.san(parsed)}({drop:+d}){mark}"
            tested += 1
            print(row)

    engine.quit()
    print(f"\n{tested} (position, budget) pairs tested")
    for rule in ("current", "proposed"):
        print(
            f"  {rule:>9}: {totals[rule]:>3} blunders of {BLUNDER_CP}+ cp, "
            f"total cp lost {losses[rule]:,.0f}"
        )


if __name__ == "__main__":
    main()
