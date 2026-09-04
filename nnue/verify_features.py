"""Prove the agent's feature extraction matches training's, exactly.

This is the highest-risk seam in the whole project. Training builds features from
a python-chess board (mirroring squares when Black is to move). The agent must
build them from sunfish's 120-char board string, which is ALREADY rotated so the
side to move reads as uppercase at the bottom. If those two disagree even
slightly, the net is fed inputs it never trained on, the evaluation becomes
noise, and nothing about it looks like a bug: the engine just plays badly.

So we check both paths agree on a few thousand random positions rather than
reasoning about it and hoping.
"""

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402

from nnue.features import features_from_board  # noqa: E402

# Sunfish piece chars in python-chess piece-type order (PAWN..KING).
PIECE_ORDER = "PNBRQK"


def features_from_sunfish(pos: "agent.Position") -> np.ndarray:
    """Active feature indices from a sunfish Position.

    The board string is already side-to-move relative: uppercase is the mover.
    """
    out = []
    for i, ch in enumerate(pos.board):
        if ch == "." or ch == " " or ch == "\n":
            continue
        upper = ch.upper()
        if upper not in PIECE_ORDER:
            continue
        piece_type = PIECE_ORDER.index(upper)
        rel_colour = 0 if ch.isupper() else 1
        square = agent.sunfish_to_square(i)
        out.append((rel_colour * 6 + piece_type) * 64 + square)
    return np.array(sorted(out), dtype=np.int32)


def random_position(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for _ in range(rng.randint(0, 60)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if board.is_game_over():
            break
    return board


def main() -> None:
    rng = random.Random(12345)
    checked = 0
    mismatches = 0
    for _ in range(3000):
        board = random_position(rng)
        if board.is_game_over():
            continue
        train_idx = np.array(sorted(features_from_board(board)), dtype=np.int32)
        pos = agent.board_to_sunfish(board)
        agent_idx = features_from_sunfish(pos)
        checked += 1
        if not np.array_equal(train_idx, agent_idx):
            mismatches += 1
            if mismatches <= 3:
                print(f"MISMATCH on {board.fen()}")
                print(f"  training: {train_idx.tolist()}")
                print(f"  agent   : {agent_idx.tolist()}")
                only_train = sorted(set(train_idx.tolist()) - set(agent_idx.tolist()))
                only_agent = sorted(set(agent_idx.tolist()) - set(train_idx.tolist()))
                print(f"  only in training: {only_train}")
                print(f"  only in agent   : {only_agent}")
    print(f"\nchecked {checked} positions, {mismatches} mismatches")
    if mismatches:
        raise SystemExit("feature encodings DISAGREE - fix before training further")
    print("OK: agent and training feature encodings are identical")


if __name__ == "__main__":
    main()
