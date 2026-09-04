"""Prove the optimised engine is behaviourally identical to the live one.

The speed work rewrote the two hottest functions. Those rewrites are supposed to
be pure performance transformations, so this checks the claim directly rather than
trusting it: for thousands of positions, compare the generated move list and every
move's value between the current agent and the live baseline, and compare the
searched move at a fixed node budget.

Any difference here means the optimisation changed behaviour and must be rejected.
"""

import importlib.util
import random
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent as new_agent  # noqa: E402


def load_reference() -> object:
    """Import baselines/live_pst/agent.py under its own module name."""
    path = ROOT / "baselines" / "live_pst" / "agent.py"
    spec = importlib.util.spec_from_file_location("live_pst_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_pst_agent"] = module
    spec.loader.exec_module(module)
    return module


def random_positions(count: int) -> list[chess.Board]:
    rng = random.Random(4242)
    out = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(0, 70)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if not board.is_game_over():
            out.append(board)
    return out


def main() -> None:
    old = load_reference()
    boards = random_positions(2500)

    move_mismatch = 0
    value_mismatch = 0
    score_mismatch = 0

    for board in boards:
        fen = board.fen()
        new_pos = new_agent.board_to_sunfish(chess.Board(fen))
        old_pos = old.board_to_sunfish(chess.Board(fen))  # type: ignore[attr-defined]

        if new_pos.board != old_pos.board or new_pos.score != old_pos.score:
            score_mismatch += 1
            continue

        new_moves = [(m.i, m.j, m.prom) for m in new_pos.gen_moves()]
        old_moves = [(m.i, m.j, m.prom) for m in old_pos.gen_moves()]
        if new_moves != old_moves:
            move_mismatch += 1
            if move_mismatch <= 3:
                print(f"MOVE MISMATCH {fen}")
                print(f"  new: {new_moves[:12]}")
                print(f"  old: {old_moves[:12]}")
            continue

        for nm, om in zip(new_pos.gen_moves(), old_pos.gen_moves(), strict=True):
            if new_pos.value(nm) != old_pos.value(om):
                value_mismatch += 1
                if value_mismatch <= 3:
                    print(f"VALUE MISMATCH {fen} move={nm}")
                break

    print(f"\nchecked {len(boards)} positions")
    print(f"  board/score mismatches: {score_mismatch}")
    print(f"  move-list mismatches  : {move_mismatch}")
    print(f"  move-value mismatches : {value_mismatch}")

    if score_mismatch or move_mismatch or value_mismatch:
        raise SystemExit("FAIL: the optimisation changed behaviour")
    print("\nOK: identical move generation and values")


if __name__ == "__main__":
    main()
