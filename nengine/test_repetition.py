"""Regression checks for threefold detection across game and search history."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.search import _is_third_repetition  # noqa: E402


def main() -> None:
    game = np.zeros(600, dtype=np.int64)
    path = np.zeros(64, dtype=np.int64)

    game[0] = 101
    assert not _is_third_repetition(101, game, 1, path, 0)

    # The root position plus an earlier node in this variation make a third hit.
    path[0] = 101
    assert _is_third_repetition(101, game, 1, path, 1)

    # Two positions supplied by the runner are already two real occurrences.
    game[1] = 202
    game[2] = 202
    assert _is_third_repetition(202, game, 3, path, 0)
    print("threefold repetition checks passed")


if __name__ == "__main__":
    main()
