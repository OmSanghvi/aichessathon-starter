"""Pick the labelling depth: quality rises with depth, throughput falls.

Training data only; the labelling binary never ships.
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnue.gen_data import UciLabeller  # noqa: E402

ENGINE = "/opt/homebrew/bin/stockfish"

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4rrk1/pp1n1ppp/2pb4/3p4/3P4/2NBP3/PP3PPP/2R2RK1 w - - 0 1",
]


def main() -> None:
    for depth in (8, 10, 12, 14):
        lab = UciLabeller(ENGINE, depth)
        boards = [chess.Board(f) for f in FENS]
        t0 = time.perf_counter()
        reps = 4
        scores = []
        for _ in range(reps):
            for b in boards:
                scores.append(lab.score(b))
        elapsed = time.perf_counter() - t0
        n = len(boards) * reps
        per = elapsed / n * 1000
        print(
            f"depth {depth:2d}: {per:6.1f} ms/pos  ({1/(elapsed/n):6.1f} pos/sec)  "
            f"1M in {(elapsed/n)*1_000_000/3600:5.1f} h   sample={scores[:4]}"
        )
        lab.close()


if __name__ == "__main__":
    main()
