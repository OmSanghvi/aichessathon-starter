"""The feature encoding. Single source of truth for training AND the agent.

Both sides import this. Last time each side had its own copy and they silently
disagreed on Black-to-move positions, which trains a net that plays like noise
while nothing looks broken. nnue2/verify_features.py asserts they match.

ENCODING: king-bucketed piece-square features.

    index = (king_bucket * 12 + relative_colour * 6 + piece_type) * 64 + square

  * king_bucket   which region OUR king occupies (KING_BUCKETS of them)
  * relative_colour  0 = side to move, 1 = opponent
  * piece_type    0..5, python-chess order (PAWN..KING) minus one
  * square        0..63 in the mover's frame

Why buckets. Plain piece-square features cannot express "this enemy rook is
dangerous BECAUSE my king is on g8" - the king's location and the rook's location
are separate inputs with no interaction. Bucketing gives the net a distinct set of
weights per king region, so it can learn placement danger conditioned on where the
king actually is. That matters here specifically: the measured failure is king
safety, and it persists at every time budget, so it is missing knowledge rather
than missing depth.

ORIENTATION: everything is from the side to move's point of view, rotated 180
degrees when Black is to move (square -> 63 - square). This matches sunfish, which
keeps every position in the mover's frame via board[::-1].swapcase(). A vertical
mirror (square ^ 56) would disagree on every Black-to-move position, and it cannot
be corrected inside the search because a sunfish Position carries no ply parity.
That exact mismatch was the v1 bug.
"""

import chess
import numpy as np

# Eight king regions: four two-file zones, each split between the back two ranks
# and the rest of the board in the mover's frame. Four regions could not tell g1
# from e1; sixteen 2x2 regions left most advanced-king regions untrained. This
# shape keeps local castling geometry while giving every feature family enough of
# the 6.6M-position data. Raw boards make the upgrade free to regenerate.
KING_BUCKETS = 8
NUM_FEATURES = KING_BUCKETS * 12 * 64  # 6144
MAX_PIECES = 32

# Labels are clamped here. Beyond this the exact number stops changing which move
# you pick, and unbounded mate scores would dominate the regression.
EVAL_CLAMP = 2000

# Board codes used by the on-disk format: 0 empty, 1..6 white P..K, 7..12 black.
EMPTY = 0


def king_bucket(king_square: int) -> int:
    """Which region the mover's king is in, in the mover's own frame."""
    file_ = king_square & 7
    rank = king_square >> 3
    advanced = 1 if rank >= 2 else 0
    return advanced * 4 + (file_ // 2)


def board_to_codes(board: chess.Board) -> tuple[np.ndarray, int]:
    """Raw storage form: 64 piece codes plus whose turn it is.

    Deliberately NOT feature indices. v1 stored indices, so redesigning the
    feature set invalidated every position already generated. Storing the board
    keeps hours of labelling reusable across feature experiments.
    """
    codes = np.zeros(64, dtype=np.int8)
    for square, piece in board.piece_map().items():
        codes[square] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    return codes, int(board.turn)


def codes_to_features(codes: np.ndarray, turn: int) -> np.ndarray:
    """Active feature indices from the stored board form.

    turn is 1 for White to move (python-chess chess.WHITE is True).
    """
    flip = turn == 0
    # Locate the mover's king first; every feature index depends on its bucket.
    # Codes are 1..6 white P..K and 7..12 black P..K, so the kings are 6 and 12.
    mover_king_code = 12 if flip else 6
    king_sq = -1
    for square in range(64):
        if codes[square] == mover_king_code:
            king_sq = 63 - square if flip else square
            break
    bucket = king_bucket(king_sq) if king_sq >= 0 else 0

    out = np.empty(MAX_PIECES, dtype=np.int32)
    count = 0
    base = bucket * 12
    for square in range(64):
        # int() matters: the stored board is int8, and index arithmetic multiplies
        # by 64, which silently overflows int8 and produces negative features.
        code = int(codes[square])
        if code == EMPTY:
            continue
        piece_type = (code - 1) % 6  # 0..5
        is_white = code <= 6
        # relative_colour: 0 for the side to move, 1 for the opponent
        rel = 0 if (is_white == (turn == 1)) else 1
        sq = 63 - square if flip else square
        out[count] = (base + rel * 6 + piece_type) * 64 + sq
        count += 1
    return out[:count]


def features_from_board(board: chess.Board) -> np.ndarray:
    codes, turn = board_to_codes(board)
    return codes_to_features(codes, turn)


def pack_boards(list_of_codes: list[np.ndarray]) -> np.ndarray:
    return np.stack(list_of_codes).astype(np.int8)
