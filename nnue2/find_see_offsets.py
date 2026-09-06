"""Find the correct enemy-pawn attack offsets on sunfish's board, empirically.

Reasoning about sunfish's rotated frame got this wrong once already, so instead of
arguing about directions this brute-forces every candidate offset pair against
python-chess and reports which pair agrees everywhere.
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402

CANDIDATES = [-11, -9, 9, 11]


def main() -> None:
    rng = random.Random(31337)
    # For each candidate offset, count how often "an enemy pawn sits at j+offset"
    # coincides with "python-chess says j is attacked by an enemy pawn".
    hits: dict[int, int] = defaultdict(int)
    truth_total = 0
    samples = []

    for _ in range(500):
        b = chess.Board()
        for _ in range(rng.randint(4, 60)):
            mv = list(b.legal_moves)
            if not mv:
                break
            b.push(rng.choice(mv))
            if b.is_game_over():
                break
        if b.is_game_over():
            continue
        pos = agent.board_to_sunfish(b)
        mover = b.turn
        for square in chess.SQUARES:
            j = agent.square_to_sunfish(square)
            truth = any(
                (p := b.piece_at(a)) is not None
                and p.piece_type == chess.PAWN
                and p.color != mover
                for a in b.attackers(not mover, square)
            )
            if truth:
                truth_total += 1
                for off in CANDIDATES:
                    k = j + off
                    if 0 <= k < 120 and pos.board[k] == "p":
                        hits[off] += 1
            samples.append((pos.board, j, truth))

    print(f"{truth_total} squares genuinely attacked by an enemy pawn\n")
    print("how often each single offset explains a real attack:")
    for off in CANDIDATES:
        print(f"  offset {off:>3}: {hits[off]:>6} ({hits[off] / max(truth_total, 1):.1%})")

    # Now test PAIRS for exact agreement (no false positives or negatives).
    print("\nexact agreement per offset pair:")
    best = None
    for a in CANDIDATES:
        for c in CANDIDATES:
            if a >= c:
                continue
            wrong = 0
            for board, j, truth in samples:
                got = (0 <= j + a < 120 and board[j + a] == "p") or (
                    0 <= j + c < 120 and board[j + c] == "p"
                )
                if got != truth:
                    wrong += 1
            rate = wrong / max(len(samples), 1)
            print(f"  ({a:>3},{c:>3}): {wrong:>6} disagreements ({rate:.3%})")
            if best is None or wrong < best[2]:
                best = (a, c, wrong)
    assert best is not None
    print(f"\nBEST PAIR: ({best[0]}, {best[1]}) with {best[2]} disagreements")
    if best[2] == 0:
        print("  -> use board[j%+d] and board[j%+d]" % (best[0], best[1]))
    else:
        print("  -> no pair is exact; a pawn-only test cannot be made reliable this way")


if __name__ == "__main__":
    main()
