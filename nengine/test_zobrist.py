"""Verify incremental Zobrist updates against full recomputation."""

import random
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.board import SQ120, encode, make_move, unmake_move  # noqa: E402
from nengine.search import (  # noqa: E402
    ZOB_CASTLE,
    ZOB_EP,
    ZOB_PIECE,
    ZOB_SIDE,
    zobrist,
    zobrist_after_move,
)
from nengine.test_perft import to_arrays  # noqa: E402


def encoded(position: chess.Board, move: chess.Move) -> int:
    flag = 1 if position.is_en_passant(move) else 2 if position.is_castling(move) else 0
    promo = 0 if move.promotion is None else move.promotion + (0 if position.turn else 6)
    return encode(int(SQ120[move.from_square]), int(SQ120[move.to_square]), promo, flag)


def check(position: chess.Board, move: chess.Move) -> None:
    board, side, castling, ep = to_arrays(position)
    before = board.copy()
    h = zobrist(board, side, castling, ep, ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP)
    core_move = encoded(position, move)
    captured, new_castling, new_ep = make_move(board, side, castling, ep, core_move)
    incremental = zobrist_after_move(
        h, board, side, castling, ep, core_move, captured, new_castling, new_ep,
        ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP,
    )
    full = zobrist(board, 1 - side, new_castling, new_ep, ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP)
    if incremental != full:
        raise SystemExit(f"hash mismatch after {move.uci()}: {incremental} != {full}")
    unmake_move(board, side, core_move, captured)
    if not np.array_equal(board, before):
        raise SystemExit(f"board did not restore after {move.uci()}")


def main() -> None:
    special = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
    ]
    for fen, uci in special:
        position = chess.Board(fen)
        check(position, chess.Move.from_uci(uci))

    rng = random.Random(2026)
    position = chess.Board()
    for _ in range(100):
        legal = list(position.legal_moves)
        if not legal:
            position = chess.Board()
            continue
        move = rng.choice(legal)
        check(position, move)
        position.push(move)
    print("incremental Zobrist checks passed")


if __name__ == "__main__":
    main()
