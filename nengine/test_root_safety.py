"""Regression tests for the root-level negative-SEE capture safety rail."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.test_search import move_to_uci, run  # noqa: E402


def main() -> None:
    queen_hang = "5r2/1p3pk1/5R2/p5PQ/P1Bq3P/1P1p4/8/4nK2 b - - 3 53"
    _rows, best = run(queen_hang, 8)
    got = move_to_uci(best)
    print(f"hanging queen: {got} (not d4c4)")
    if got == "d4c4":
        raise SystemExit("FAIL catastrophic guard allowed Qxc4")

    # The guard must remain narrow: this -220 SEE knight sacrifice is below
    # the threshold and should preserve the prior engine's candidate choice.
    knight_sac = "1rbr2k1/1p3ppp/3pq3/pB1N4/1nP1p1n1/5N1P/PP3PP1/R2QR1K1 b - - 0 20"
    _rows, best = run(knight_sac, 8)
    got = move_to_uci(best)
    print(f"ordinary sacrifice: {got} (expect g4f2)")
    if got != "g4f2":
        raise SystemExit(f"FAIL narrow guard changed Nxf2 choice to {got}")
    print("Catastrophic-capture safety regression checks passed")


if __name__ == "__main__":
    main()
