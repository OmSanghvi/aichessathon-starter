"""Where does per-node time actually go?

Three measurements have now agreed that the engine loses games to being too
shallow in endgames rather than to bad judgement, so the question is what a node
costs and which function to attack. Guessing has already cost two failed changes,
so this profiles the real search on real positions.

Reports cProfile's cumulative and per-call costs, plus a node-rate breakdown per
position type, since endgames (where we lose) may profile differently from
middlegames.
"""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

POSITIONS = {
    "opening": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "middlegame": "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "complex mid": "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7",
    "the lost endgame": "6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42",
    "rook endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}

THINK_S = 2.0


def run_search(fen: str, think_s: float = THINK_S) -> tuple[int, int]:
    """Search one position for a fixed wall time; return (nodes, depth reached)."""
    pos = agent.board_to_sunfish(chess.Board(fen))
    searcher = agent.Searcher()
    searcher.soft = searcher.deadline = time.time() + think_s
    depth = 0
    try:
        for d, _gamma, _score, _move in searcher.search([pos]):
            depth = max(depth, d)
    except agent.Stop:
        pass
    return searcher.nodes, depth


def node_rates() -> None:
    print("node rate and depth reached per position type\n")
    print(f"{'position':<20}{'nodes':>10}{'nodes/sec':>12}{'us/node':>10}{'depth':>7}")
    for name, fen in POSITIONS.items():
        nodes, depth = run_search(fen)
        nps = nodes / THINK_S
        print(f"{name:<20}{nodes:>10,}{nps:>12,.0f}{1e6 / nps:>10.2f}{depth:>7}")


def profile_hot_path() -> None:
    """Profile a mix of positions so the numbers are not one-position artefacts."""
    profiler = cProfile.Profile()
    profiler.enable()
    for fen in POSITIONS.values():
        run_search(fen, 1.5)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("tottime")
    stats.print_stats(18)
    print("\n\n=== profile, sorted by time IN the function (tottime) ===")
    print(stream.getvalue())


def main() -> None:
    node_rates()
    profile_hot_path()


if __name__ == "__main__":
    main()
