"""Assert the residual-training anchor is bit-identical to the shipped evaluator."""

import random
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
from nnue2.features import board_to_codes  # noqa: E402
from nnue2.nengine_pst import score_codes, scores_batch  # noqa: E402


def main() -> None:
    rng = random.Random(20260905)
    boards = []
    turns = []
    expected = []
    for _ in range(500):
        board = chess.Board()
        for _ in range(rng.randrange(80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if board.is_game_over():
            continue
        codes, turn = board_to_codes(board)
        mailbox, side, _, _ = agent._to_arrays(board)
        expected.append(agent.evaluate(
            mailbox, side, agent.PST, agent.PST_KING_MID, agent.PST_KING_END))
        boards.append(codes)
        turns.append(turn)

    import numpy as np

    packed = np.stack(boards).astype(np.int8)
    turn_array = np.array(turns, dtype=np.int8)
    got = scores_batch(packed, turn_array)
    if not np.array_equal(got, np.array(expected, dtype=np.int32)):
        index = int(np.flatnonzero(got != np.array(expected))[0])
        raise SystemExit(f"anchor mismatch at position {index}: {got[index]} != {expected[index]}")
    # Exercise the one-position JIT entry point too.
    assert score_codes(packed[0], turn_array[0]) == expected[0]
    print(f"OK: nengine residual anchor matches the shipped evaluator on {len(boards)} positions")


if __name__ == "__main__":
    main()
