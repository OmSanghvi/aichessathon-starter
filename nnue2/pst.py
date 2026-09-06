"""Sunfish's incremental piece-square score, computed from the stored board.

Needed to train on the RESIDUAL. The diagnosis was that the net is a better
evaluation than the tables (correlation 0.810 against 0.758) but disagrees with
them by 311 cp on average, while the search prunes using the table score and only
scores leaves with the net. Those two decisions pull apart, and a better net cannot
fix that.

Training on `label - pst_score` and evaluating as `pos.score + correction` keeps the
evaluation anchored where sunfish's tuned margins (QS_A, NULL_MARGIN, the calmness
test) remain valid, while the net supplies the positional knowledge the tables lack.

This must reproduce agent.board_to_sunfish().score exactly; verify_pst() checks it.
Computing it from the stored boards means the 6.6M already-labelled positions are
reusable and nothing has to be generated again.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402

PIECE_CHARS = "PNBRQK"


def _square_to_sunfish(square: int) -> int:
    return 91 + (square & 7) - 10 * (square >> 3)


def pst_score_from_codes(codes: np.ndarray, turn: int) -> int:
    """Sunfish's pos.score for the stored board, from the mover's point of view."""
    score = 0
    white_to_move = turn == 1
    for square in range(64):
        code = int(codes[square])
        if code == 0:
            continue
        is_white = code <= 6
        ch = PIECE_CHARS[(code - 1) % 6]
        if white_to_move:
            sq, own = square, is_white
        else:
            # Mover frame is a 180 degree rotation with colours swapped.
            sq, own = 63 - square, not is_white
        i = _square_to_sunfish(sq)
        if own:
            score += agent.pst[ch][i]
        else:
            score -= agent.pst[ch][119 - i]
    return score


def pst_scores_batch(boards: np.ndarray, turns: np.ndarray) -> np.ndarray:
    out = np.empty(boards.shape[0], dtype=np.int32)
    for r in range(boards.shape[0]):
        out[r] = pst_score_from_codes(boards[r], int(turns[r]))
    return out


def verify_pst(n: int = 600) -> None:
    """Assert this matches the agent's own score on real positions."""
    import random

    import chess

    from nnue2.features import board_to_codes

    # The king table is phase-dependent in sunfish and is set per search; pin it to
    # the middlegame table so both sides of the comparison use the same one.
    agent.pst["K"] = agent.K_MID

    rng = random.Random(99)
    checked = bad = 0
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(0, 60)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
            if b.is_game_over():
                break
        if b.is_game_over():
            continue
        codes, turn = board_to_codes(b)
        mine = pst_score_from_codes(codes, turn)
        theirs = agent.board_to_sunfish(b).score
        checked += 1
        if mine != theirs:
            bad += 1
            if bad <= 3:
                print(f"  MISMATCH {b.fen()}: mine {mine} vs agent {theirs}")
    print(f"checked {checked} positions, {bad} mismatches")
    if bad:
        raise SystemExit("FAIL: pst score does not match the agent")
    print("OK: pst score matches agent.board_to_sunfish exactly")


if __name__ == "__main__":
    verify_pst()
