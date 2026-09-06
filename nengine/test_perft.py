"""Perft: the correctness gate for the numba move generator.

Perft counts legal leaf nodes at a given depth. It is the standard test because it
catches every class of move-generation bug - castling through check, en-passant
legality, promotion counts, pin handling - in one number. If perft matches
python-chess on the tricky positions, the generator is right.

Known-good values come from python-chess itself, so there is no chance of copying a
wrong reference table.
"""

import sys
import time
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.board import (  # noqa: E402
    CR_BK,
    CR_BQ,
    CR_WK,
    CR_WQ,
    EMPTY,
    IS_SLIDER,
    N_OFFSETS,
    OFF,
    OFFSETS,
    SQ120,
    perft,
)

# The standard perft suite: start position plus the positions that historically
# break generators (Kiwipete for castling/pins, position 3 for en passant).
CASES = [
    ("startpos", chess.STARTING_FEN, 4),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3),
    ("promotions", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 4),
]

PIECE_TO_CODE = {
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9, (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
}


def to_arrays(b: chess.Board):
    board = np.full(120, OFF, dtype=np.int8)
    for s in range(64):
        board[SQ120[s]] = EMPTY
    for sq, piece in b.piece_map().items():
        board[SQ120[sq]] = PIECE_TO_CODE[(piece.piece_type, piece.color)]
    side = 0 if b.turn == chess.WHITE else 1
    cr = 0
    if b.has_kingside_castling_rights(chess.WHITE):
        cr |= CR_WK
    if b.has_queenside_castling_rights(chess.WHITE):
        cr |= CR_WQ
    if b.has_kingside_castling_rights(chess.BLACK):
        cr |= CR_BK
    if b.has_queenside_castling_rights(chess.BLACK):
        cr |= CR_BQ
    ep = int(SQ120[b.ep_square]) if b.ep_square is not None else 0
    return board, side, cr, ep


def ref_perft(b: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for m in b.legal_moves:
        b.push(m)
        total += ref_perft(b, depth - 1)
        b.pop()
    return total


def main() -> None:
    print("compiling (first call includes numba compilation)...")
    b0, s0, c0, e0 = to_arrays(chess.Board())
    t0 = time.time()
    perft(b0.copy(), s0, c0, e0, 1, OFFSETS, N_OFFSETS, IS_SLIDER)
    print(f"  compiled in {time.time() - t0:.1f}s\n")

    failures = 0
    for name, fen, depth in CASES:
        b = chess.Board(fen)
        board, side, cr, ep = to_arrays(b)
        for d in range(1, depth + 1):
            t0 = time.time()
            got = perft(board.copy(), side, cr, ep, d, OFFSETS, N_OFFSETS, IS_SLIDER)
            dt = time.time() - t0
            want = ref_perft(chess.Board(fen), d)
            ok = got == want
            if not ok:
                failures += 1
            nps = got / dt if dt > 0 else 0
            print(f"  {'ok ' if ok else 'FAIL'} {name:<11} depth {d}: "
                  f"got {got:>9,}  want {want:>9,}  "
                  f"{dt:>6.2f}s  {nps:>10,.0f} nodes/sec")
        print()

    if failures:
        raise SystemExit(f"FAILED: {failures} perft mismatches - generator is wrong")
    print("ALL PERFT CORRECT: the move generator is sound")


if __name__ == "__main__":
    main()
