"""Pre-ship safety check, focused on the failure that nearly happened in rated play.

The live version consumed 166.8s of 168.0s available in a 96-move game, finishing
with 1.2s. A flag is an automatic loss, so the single most important thing to
confirm before uploading is that a long game keeps a real buffer.

Simulates a full game against a fixed clock exactly as the referee accounts for it:
the agent is handed its remaining clock, we charge it the wall time it actually
spends, then credit the increment.
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

BASE_MS = 120_000
INCREMENT_MS = 500
PLY_CAP = 300


def main() -> None:
    board = chess.Board()
    clock = float(BASE_MS)
    our_moves = 0
    slowest = 0.0
    fastest = 1e9
    total = 0.0
    min_clock = clock

    # Play both sides with our own engine so the game runs long and stays legal.
    while len(board.move_stack) < PLY_CAP and not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            t0 = time.time()
            uci = agent.get_move(board.fen(), int(clock))
            spent = (time.time() - t0) * 1000
            clock -= spent
            if clock < 0:
                print(f"FLAGGED at our move {our_moves + 1}")
                return
            min_clock = min(min_clock, clock)
            clock += INCREMENT_MS
            our_moves += 1
            slowest = max(slowest, spent)
            fastest = min(fastest, spent)
            total += spent
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                print(f"ILLEGAL MOVE {uci} at our move {our_moves}")
                return
            board.push(move)
            if our_moves % 15 == 0:
                print(f"  our move {our_moves:>3} (ply {len(board.move_stack):>3}): "
                      f"spent {spent / 1000:>5.2f}s  clock {clock / 1000:>6.1f}s")
        else:
            # Opponent: cheap legal reply so the game progresses.
            board.push(next(iter(board.legal_moves)))

    available = BASE_MS + our_moves * INCREMENT_MS
    print(f"\ngame ended after {len(board.move_stack)} plies "
          f"({our_moves} of our moves), {board.result(claim_draw=True)}")
    print(f"  time used   : {total / 1000:.1f}s of {available / 1000:.1f}s available")
    print(f"  clock left  : {clock / 1000:.1f}s")
    print(f"  MIN clock   : {min_clock / 1000:.1f}s   <-- flag margin")
    print(f"  slowest move: {slowest / 1000:.2f}s")
    print(f"  fastest move: {fastest / 1000:.2f}s")
    print(f"  average     : {total / max(our_moves, 1) / 1000:.2f}s")
    print("\nlive version for comparison: 96 moves, 166.8s of 168.0s, 1.2s left")
    if min_clock < 5000:
        print("WARNING: flag margin under 5s")
    else:
        print("OK: comfortable flag margin")


if __name__ == "__main__":
    main()
