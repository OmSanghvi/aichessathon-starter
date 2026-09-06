"""Feature-order and integer-forward checks for the C++ NNUE candidate."""

import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.board import (  # noqa: E402
    IS_SLIDER,
    N_OFFSETS,
    OFFSETS,
    SQ120,
    gen_moves,
    make_move,
    mv_from,
    mv_to,
    unmake_move,
)
from nengine.cpp_nnue import (  # noqa: E402
    apply_move,
    evaluate_full,
    load_network,
    refresh_accumulator,
    undo_move,
)
from nengine.test_perft import to_arrays  # noqa: E402

NETWORK = ROOT / "weights" / "cpp_nnue.bin"


def main() -> None:
    ft, bias, out, out_bias = load_network(NETWORK)
    # Values independently queried from the team's C++ binary with ``eval``.
    cases = [
        (chess.STARTING_FEN, 32),
        ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", 1),
        ("6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42", -93),
    ]
    for fen, expected in cases:
        board, side, _castling, _ep = to_arrays(chess.Board(fen))
        actual = evaluate_full(board, side, ft, bias, out, int(out_bias))
        print(f"{fen[:38]:<38} {actual:+d}")
        if actual != expected:
            raise SystemExit(f"FAIL expected {expected:+d}, got {actual:+d}")

    # Normal move, castling, en passant, and promotion must leave the incremental
    # accumulator identical to a full refresh, then restore exactly on undo.
    deltas = [
        (chess.STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6"),
        ("7k/P7/8/8/8/8/8/K7 w - - 0 1", "a7a8q"),
    ]
    for fen, uci in deltas:
        position = chess.Board(fen)
        arr, side, castling, ep = to_arrays(position)
        before = arr.copy()
        move = chess.Move.from_uci(uci)
        # The mailbox generator encodes flags, so use its legal representation.
        moves = np.empty(256, dtype=np.int32)
        count = gen_moves(arr, side, castling, ep, moves, OFFSETS, N_OFFSETS, IS_SLIDER)
        core = next(
            int(moves[i])
            for i in range(count)
            if mv_from(moves[i]) == int(SQ120[move.from_square])
            and mv_to(moves[i]) == int(SQ120[move.to_square])
        )
        mover = arr[mv_from(core)]
        captured, _new_castling, _new_ep = make_move(arr, side, castling, ep, core)
        placed = arr[mv_to(core)]
        acc = refresh_accumulator(before, ft, bias)
        apply_move(acc, mover, placed, captured, side, core, ft)
        if not np.array_equal(acc, refresh_accumulator(arr, ft, bias)):
            raise SystemExit(f"FAIL accumulator update {uci}")
        undo_move(acc, mover, placed, captured, side, core, ft)
        unmake_move(arr, side, core, captured)
        if (
            not np.array_equal(acc, refresh_accumulator(arr, ft, bias))
            or not np.array_equal(arr, before)
        ):
            raise SystemExit(f"FAIL accumulator undo {uci}")
    print("incremental accumulator updates match full refresh")
    print("C++ NNUE reference checks passed")


if __name__ == "__main__":
    main()
