"""Sanity-check generated shards: shapes, ranges, label distribution."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnue.features import EVAL_CLAMP, MAX_PIECES, NUM_FEATURES  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    shards = sorted(DATA_DIR.glob("shard_*.npz"))
    if not shards:
        raise SystemExit("no shards in nnue/data")
    total = 0
    all_labels = []
    for path in shards:
        with np.load(path) as z:
            feats, labels = z["features"], z["labels"]
        assert feats.shape[1] == MAX_PIECES, feats.shape
        assert feats.shape[0] == labels.shape[0], (feats.shape, labels.shape)
        valid = feats[feats >= 0]
        assert valid.min() >= 0 and valid.max() < NUM_FEATURES, (valid.min(), valid.max())
        assert abs(int(labels.min())) <= EVAL_CLAMP and abs(int(labels.max())) <= EVAL_CLAMP
        pieces = (feats >= 0).sum(axis=1)
        total += feats.shape[0]
        all_labels.append(labels)
        print(
            f"{path.name}: rows={feats.shape[0]:,} pieces/pos min={pieces.min()} "
            f"max={pieces.max()} mean={pieces.mean():.1f}"
        )
    labels = np.concatenate(all_labels)
    print(f"\ntotal rows: {total:,}")
    print(f"label cp: min={labels.min()} max={labels.max()} mean={labels.mean():.1f} "
          f"std={labels.std():.1f}")
    for q in (1, 10, 25, 50, 75, 90, 99):
        print(f"  p{q:<2d} = {np.percentile(labels, q):8.0f}")
    # A healthy set is not all one-sided and not all zeros.
    frac_extreme = float((np.abs(labels) >= EVAL_CLAMP).mean())
    print(f"fraction at the clamp: {frac_extreme:.1%}")


if __name__ == "__main__":
    main()
