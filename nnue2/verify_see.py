"""Verify the cheap "is this square defended by an enemy pawn" test.

Sunfish orders captures by pos.value(), which is essentially most-valuable-victim
and ignores the attacker and whether the target is defended. So a queen capturing a
pawn that a pawn recaptures scores about +100 and is searched in quiescence as if it
won material. Skipping those is standard practice - the chess programming wiki notes
quiescence "is vulnerable to search explosions in the absence of any move ordering"
- and it is the cheapest way to stop wasting nodes on losing captures.

The geometry is easy to get backwards, so it is checked against python-chess rather
than reasoned about. In sunfish's mover-relative frame our pawns move N (-10) and
capture to -11 and -9; enemy pawns move the other way, so an enemy pawn at k attacks
k+11 and k+9, which means square j is pawn-defended when board[j-11] or board[j-9]
holds a lowercase p.
"""

import random
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402


def pawn_defended_sunfish(board: str, j: int) -> bool:
    """Cheap test on the 120-char board: is j attacked by an enemy pawn?"""
    return board[j - 11] == "p" or board[j - 9] == "p"


def main() -> None:
    rng = random.Random(4242)
    checked = wrong = 0
    examples = []
    for _ in range(1500):
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
        # For every square, compare the cheap test against python-chess truth.
        for square in chess.SQUARES:
            i = agent.square_to_sunfish(square)
            # python-chess: is this square attacked by an enemy PAWN?
            truth = any(
                b.piece_at(a) is not None
                and b.piece_at(a).piece_type == chess.PAWN
                and b.piece_at(a).color != mover
                for a in b.attackers(not mover, square)
            )
            got = pawn_defended_sunfish(pos.board, i)
            checked += 1
            if truth != got:
                wrong += 1
                if len(examples) < 5:
                    examples.append((b.fen(), chess.square_name(square), truth, got))

    print(f"checked {checked:,} (position, square) pairs")
    print(f"disagreements: {wrong} ({wrong / max(checked, 1):.3%})")
    for fen, sq, truth, got in examples:
        print(f"  {fen}  square {sq}: truth {truth} got {got}")
    if wrong:
        raise SystemExit("FAIL: the pawn-defender test is wrong; do not use it")
    print("OK: cheap pawn-defender test matches python-chess exactly")


if __name__ == "__main__":
    main()
