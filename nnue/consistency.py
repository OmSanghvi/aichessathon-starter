"""Does the engine pick the same move as the clock changes?

Two different failures look identical from outside ("it played a worse move when
there was an obvious one"):

  * depth starvation - with a small budget it only reaches depth 5 or 6 and
    genuinely concludes the wrong move. More time fixes this.
  * unstable selection - the move flips around as the budget changes, including
    getting WORSE with more time. That is not a knowledge problem and more time
    does not fix it.

A move that improves monotonically with time is the first kind. A move that
oscillates is the second. This sweeps a range of clocks per position and reports
which pattern each position shows.

NOTE: if an arena is running concurrently the absolute depths will be depressed by
CPU contention, but the monotonic-vs-oscillating pattern still shows.
"""

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

# (fen, best move, description). All have a move a decent engine should find.
CASES = [
    ("6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42", "Be4",
     "the losing game's blunder (Kd4 loses 448cp)"),
    ("rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 1", "fxg4",
     "free queen on g4"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "Ra8#", "mate in 1"),
    ("8/1P6/8/8/8/2k5/8/6K1 w - - 0 1", "b8=Q", "promote to queen"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", None,
     "quiet position, just check stability"),
    ("8/8/8/3k4/8/8/3r4/3K4 w - - 0 1", "Kxd2", "capture the hanging rook"),
]

CLOCKS = (1_000, 2_000, 3_000, 5_000, 8_000, 12_000, 20_000, 40_000, 80_000, 120_000)


def main() -> None:
    print(f"{'position':<34}" + "".join(f"{c // 1000:>6}s" for c in CLOCKS))
    unstable = 0
    for fen, want, label in CASES:
        board = chess.Board(fen)
        row = f"{label[:32]:<34}"
        picks = []
        for clock in CLOCKS:
            uci = agent.get_move(fen, clock)
            try:
                move = chess.Move.from_uci(uci)
                san = board.san(move) if move in board.legal_moves else "ILL"
            except ValueError:
                san = "ILL"
            picks.append(san)
            row += f"{san:>7}"
        print(row)

        distinct = len(set(picks))
        if want is not None:
            correct = [p == want for p in picks]
            # Monotonic means: once right, stays right as time grows.
            first_right = correct.index(True) if True in correct else None
            flipped_back = (
                first_right is not None and not all(correct[first_right:])
            )
            verdict = (
                "never found" if first_right is None
                else "OSCILLATES (gets worse with more time)" if flipped_back
                else f"monotonic, needs >={CLOCKS[first_right] // 1000}s"
            )
            if flipped_back:
                unstable += 1
            print(f"{'':<34}  want {want}: {verdict}")
        else:
            print(f"{'':<34}  {distinct} distinct moves across budgets")
    print(f"\npositions that get WORSE with more time: {unstable}")
    print("Those are selection instability, not depth. Time alone will not fix them.")


if __name__ == "__main__":
    main()
