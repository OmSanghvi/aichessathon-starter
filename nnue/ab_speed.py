"""Interleaved A/B node-rate comparison: live baseline vs optimised.

Single-shot timings on this machine swung by 20% run to run, which is enough to
invent or hide a 6% effect. So both versions are measured alternately, several
times each, and reported as a median with the spread, on a fixed NODE budget so
the comparison is time-per-node rather than nodes-per-time.
"""

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent as new_agent  # noqa: E402

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7",
    "6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]

THINK_S = 1.2
REPEATS = 5


def load_reference() -> object:
    path = ROOT / "baselines" / "live_pst" / "agent.py"
    spec = importlib.util.spec_from_file_location("live_pst_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_pst_agent"] = module
    spec.loader.exec_module(module)
    return module


def measure(module: object) -> float:
    """Total nodes searched across all positions in a fixed wall time each."""
    total = 0
    for fen in POSITIONS:
        pos = module.board_to_sunfish(chess.Board(fen))  # type: ignore[attr-defined]
        searcher = module.Searcher()  # type: ignore[attr-defined]
        searcher.soft = searcher.deadline = time.time() + THINK_S
        try:
            for _ in searcher.search([pos]):
                pass
        except module.Stop:  # type: ignore[attr-defined]
            pass
        total += searcher.nodes
    return total


def main() -> None:
    old = load_reference()
    old_runs: list[float] = []
    new_runs: list[float] = []

    print(f"interleaving {REPEATS} repeats, {THINK_S}s per position, "
          f"{len(POSITIONS)} positions\n")
    for r in range(REPEATS):
        o = measure(old)
        n = measure(new_agent)
        old_runs.append(o)
        new_runs.append(n)
        print(f"  repeat {r + 1}: live={o:>9,.0f} nodes   optimised={n:>9,.0f} nodes "
              f"({(n / o - 1) * 100:+.1f}%)")

    om, nm = statistics.median(old_runs), statistics.median(new_runs)
    print(f"\nlive      median {om:>11,.0f} nodes  (min {min(old_runs):,.0f} "
          f"max {max(old_runs):,.0f})")
    print(f"optimised median {nm:>11,.0f} nodes  (min {min(new_runs):,.0f} "
          f"max {max(new_runs):,.0f})")
    print(f"\nmedian speedup: {(nm / om - 1) * 100:+.1f}%")
    spread = (max(old_runs) - min(old_runs)) / om * 100
    print(f"baseline run-to-run spread: {spread:.1f}%  "
          f"(effects smaller than this are not measurable this way)")


if __name__ == "__main__":
    main()
