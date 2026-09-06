"""Prove the agent's feature extraction matches training's, exactly.

The highest-risk seam in the project, and the one that actually broke in v1.
Training builds features from a python-chess board; the agent must build them from
sunfish's 120-char board string, which is ALREADY rotated so the mover reads as
uppercase. In v1 training used a vertical mirror (square ^ 56) while sunfish rotates
180 degrees, so 1516 of 2978 positions disagreed - every Black-to-move one - and
nothing looked broken. The net was simply fed inputs it never trained on.

Run this after ANY change to features.py. If it fails, stop; a net trained against
a mismatched encoding is worthless no matter how good its validation loss looks.
"""

import random
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
from nnue2.features import KING_BUCKETS, features_from_board, king_bucket  # noqa: E402

# Sunfish piece characters in python-chess piece-type order.
PIECE_ORDER = "PNBRQK"


def features_from_sunfish(pos: "agent.Position") -> np.ndarray:
    """Active features from a sunfish Position, as the agent must compute them.

    The board string is already mover-relative: uppercase is the side to move, and
    the frame is sunfish's 180 degree rotation.
    """
    # Find the mover's king first; the bucket depends on it.
    king_sq = -1
    for i, ch in enumerate(pos.board):
        if ch == "K":
            king_sq = agent.sunfish_to_square(i)
            break
    bucket = king_bucket(king_sq) if king_sq >= 0 else 0

    out = []
    base = bucket * 12
    for i, ch in enumerate(pos.board):
        upper = ch.upper()
        if upper not in PIECE_ORDER:
            continue
        piece_type = PIECE_ORDER.index(upper)
        rel = 0 if ch.isupper() else 1
        square = agent.sunfish_to_square(i)
        out.append((base + rel * 6 + piece_type) * 64 + square)
    return np.array(sorted(out), dtype=np.int32)


def main() -> None:
    rng = random.Random(20260904)
    checked = mismatched = 0
    buckets_seen = set()
    for _ in range(4000):
        board = chess.Board()
        for _ in range(rng.randint(0, 70)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if board.is_game_over():
            continue

        train_idx = np.array(sorted(features_from_board(board)), dtype=np.int32)
        pos = agent.board_to_sunfish(board)
        agent_idx = features_from_sunfish(pos)
        checked += 1
        if train_idx.size:
            buckets_seen.add(int(train_idx[0]) // (12 * 64))

        if not np.array_equal(train_idx, agent_idx):
            mismatched += 1
            if mismatched <= 3:
                print(f"MISMATCH {board.fen()}")
                only_t = sorted(set(train_idx.tolist()) - set(agent_idx.tolist()))
                only_a = sorted(set(agent_idx.tolist()) - set(train_idx.tolist()))
                print(f"  only in training: {only_t[:10]}")
                print(f"  only in agent   : {only_a[:10]}")

    print(f"\nchecked {checked} positions, {mismatched} mismatches")
    print(f"king buckets exercised: {sorted(buckets_seen)} of {list(range(KING_BUCKETS))}")
    if mismatched:
        raise SystemExit("FAIL: encodings disagree - fix before training")
    if len(buckets_seen) < KING_BUCKETS:
        print("NOTE: not every bucket appeared; add positions if a bucket is untested")
    print("OK: agent and training feature encodings are identical")


if __name__ == "__main__":
    main()
