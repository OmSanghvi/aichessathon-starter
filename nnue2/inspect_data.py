"""Check the generated data before spending a night training on it.

Looks for the specific ways this dataset went wrong in v1:
  * too few positions (500k plateaued at 93 cp and lost 12.5%)
  * labels piled up at the clamp, meaning a set full of already-decided positions
    that teaches nothing about the balanced middlegames the engine actually plays
  * a game-phase skew - v1 averaged 14 pieces a position, far into the endgame
  * king buckets unevenly covered, which would leave part of the net untrained
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nnue2.features import EVAL_CLAMP, KING_BUCKETS, codes_to_features  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA))
    args = parser.parse_args()
    shards = sorted(Path(args.data_dir).glob("shard_*.npz"))
    if not shards:
        raise SystemExit("no shards in nnue2/data")

    labels_all, pieces_all, bucket_counts = [], [], np.zeros(KING_BUCKETS, dtype=np.int64)
    total = 0
    for path in shards:
        with np.load(path) as z:
            boards, turns, labels = z["boards"], z["turns"], z["labels"]
        total += labels.shape[0]
        labels_all.append(labels)
        pieces_all.append((boards != 0).sum(axis=1))
        # Sample buckets rather than every row, to keep this quick on big sets.
        step = max(1, boards.shape[0] // 4000)
        for i in range(0, boards.shape[0], step):
            idx = codes_to_features(boards[i], int(turns[i]))
            if idx.size:
                bucket_counts[int(idx[0]) // (12 * 64)] += 1
        print(f"{path.name}: {labels.shape[0]:,} rows")

    labels = np.concatenate(labels_all)
    pieces = np.concatenate(pieces_all)

    print(f"\ntotal positions: {total:,}")
    if total < 2_000_000:
        print("  WARNING: v1 had 500k and lost. 2M+ is the target.")

    print(f"\nlabel cp: mean {labels.mean():.1f}  std {labels.std():.1f}  "
          f"min {labels.min()}  max {labels.max()}")
    for q in (1, 10, 25, 50, 75, 90, 99):
        print(f"    p{q:<2} {np.percentile(labels, q):8.0f}")
    at_clamp = float((np.abs(labels) >= EVAL_CLAMP).mean())
    print(f"  fraction at the clamp: {at_clamp:.2%}"
          f"{'   <-- too many decided positions' if at_clamp > 0.05 else ''}")

    print(f"\npieces per position: mean {pieces.mean():.1f}  "
          f"min {pieces.min()}  max {pieces.max()}")
    if pieces.mean() < 16:
        print("  WARNING: endgame-skewed; raise --min-pieces or lower --max-plies")

    share = bucket_counts / max(bucket_counts.sum(), 1)
    print("\nking-bucket coverage (mover-relative regions):")
    for b in range(KING_BUCKETS):
        print(f"    bucket {b}: {share[b]:6.1%}")
    if share.min() < 0.02:
        print("  WARNING: a bucket is nearly unseen; that part of the net "
              "will be untrained")


if __name__ == "__main__":
    main()
