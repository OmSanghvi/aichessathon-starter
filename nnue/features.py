"""The feature encoding, shared by training and the agent.

This module is the single source of truth for how a position becomes network
inputs. Training and inference MUST agree exactly; a silent mismatch here is the
classic way a net that looked good in training plays like noise in a game, so
both sides import this rather than each rolling their own.

Encoding: 768 binary features = 2 relative colours x 6 piece types x 64 squares.

Everything is from the SIDE TO MOVE's point of view, so the net only ever sees
"me" and "them" and never has to learn the two sides separately.

    index = (relative_colour * 6 + piece_type) * 64 + square

with relative_colour 0 = side to move, 1 = opponent, and piece_type 0..5 in
python-chess order (PAWN..KING) minus one.

ORIENTATION, and why it is a 180 degree rotation rather than a vertical mirror:
the agent evaluates sunfish `Position` objects from inside the search, and
sunfish keeps every position in the mover's frame by calling
`board[::-1].swapcase()`, which rotates 180 degrees - it flips rank AND file. A
vertical mirror (`square ^ 56`) would disagree with that on every Black-to-move
position, and there is no way to correct for it per node because a Position in
mover frame carries no ply parity. So training adopts sunfish's convention:

    Black to move -> square 63 - square   (flip rank and file)

The frame is self-consistent for both colours, which is all the net needs. It
does mean "kingside" and "queenside" swap for Black, but consistently, so the net
simply learns in that frame. nnue/verify_features.py asserts the two paths agree.
"""

import chess
import numpy as np

NUM_FEATURES = 768
MAX_PIECES = 32

# Label clamp, centipawns. Beyond this the exact number stops mattering for
# move choice and unbounded mate scores would dominate the regression loss.
EVAL_CLAMP = 2000


def features_from_board(board: chess.Board) -> np.ndarray:
    """Return the active feature indices for ``board``, side-to-move relative."""
    mover = board.turn
    flip = mover == chess.BLACK
    out = np.empty(MAX_PIECES, dtype=np.int32)
    count = 0
    for square, piece in board.piece_map().items():
        # 180 degree rotation, matching sunfish's board[::-1].swapcase().
        sq = 63 - square if flip else square
        rel_colour = 0 if piece.color == mover else 1
        idx = (rel_colour * 6 + (piece.piece_type - 1)) * 64 + sq
        out[count] = idx
        count += 1
    return out[:count]


def features_to_dense(indices: np.ndarray) -> np.ndarray:
    """Expand active indices into a dense float32 vector, for training."""
    dense = np.zeros(NUM_FEATURES, dtype=np.float32)
    dense[indices] = 1.0
    return dense


def pack_batch(list_of_indices: list[np.ndarray]) -> np.ndarray:
    """Pack variable-length index lists into a padded int32 matrix.

    Padding uses -1 so the consumer can skip it. Storing indices rather than
    dense 768-wide rows keeps the dataset ~24x smaller on disk.
    """
    rows = len(list_of_indices)
    packed = np.full((rows, MAX_PIECES), -1, dtype=np.int32)
    for r, idx in enumerate(list_of_indices):
        packed[r, : len(idx)] = idx
    return packed
