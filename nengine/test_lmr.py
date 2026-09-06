"""Regression checks for the dynamic late-move-reduction schedule."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.search import late_move_reduction  # noqa: E402


def main() -> None:
    cases = [
        (2, 20, 0),
        (8, 3, 0),
        (3, 4, 1),
        (6, 7, 1),
        (10, 15, 1),
    ]
    for depth, move_number, expected in cases:
        actual = late_move_reduction(depth, move_number)
        if actual != expected:
            raise SystemExit(
                f"depth={depth}, move={move_number}: {actual}, expected {expected}"
            )
    print("late-move-reduction schedule checks passed")


if __name__ == "__main__":
    main()
