"""Train the NNUE-style evaluation.

Architecture: 768 -> ACC -> 32 -> 1, clipped-ReLU throughout. Deliberately small
and quantisation-friendly, because inference speed decides whether the net is
worth using at all: the benchmark said a numba int32 forward pass costs ~4.5us
against a ~30us search node, and that only holds for a net this size.

Two details that exist for quantisation rather than accuracy:
  * clipped ReLU (clamp to [0, 1]) matches the int8/int16 kernel's saturation,
    so the quantised net behaves like the trained one instead of diverging at
    the extremes.
  * weights are penalised toward a range that survives int16 scaling; a net
    trained without that constraint quantises into noise.

The target is centipawns, scaled down for a stable regression. Sunfish's pruning
constants (QS, QS_A, LMR, NULL_MARGIN, the calmness test) all live in centipawn
space, so the net has to speak the same units as the table it replaces.

Run: .venv/bin/python nnue/train.py --epochs 12 --acc 256
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnue.features import NUM_FEATURES  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"

# Centipawns are divided by this before the loss, so targets land near unit
# scale. The agent multiplies back, so the net still speaks centipawns.
CP_SCALE = 400.0


class ShardDataset(Dataset):
    """Positions stored as active feature indices, expanded to dense on demand.

    Keeping the on-disk form sparse (32 indices) rather than dense (768 floats)
    is a ~24x saving; expanding per item costs little next to the backward pass.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.features[index]
        idx = idx[idx >= 0]
        dense = torch.zeros(NUM_FEATURES, dtype=torch.float32)
        dense[torch.from_numpy(idx.astype(np.int64))] = 1.0
        target = torch.tensor([self.labels[index] / CP_SCALE], dtype=torch.float32)
        return dense, target


class NNUE(nn.Module):
    def __init__(self, acc: int = 256, hidden: int = 32) -> None:
        super().__init__()
        self.ft = nn.Linear(NUM_FEATURES, acc)
        self.l1 = nn.Linear(acc, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clipped ReLU mirrors the integer kernel's saturation.
        x = torch.clamp(self.ft(x), 0.0, 1.0)
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        return self.l2(x)


def load_data(limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    shards = sorted(DATA_DIR.glob("shard_*.npz"))
    if not shards:
        raise SystemExit("no shards in nnue/data - run gen_data.py first")
    feats, labels = [], []
    total = 0
    for path in shards:
        with np.load(path) as z:
            feats.append(z["features"])
            labels.append(z["labels"])
        total += feats[-1].shape[0]
        if limit is not None and total >= limit:
            break
    features = np.concatenate(feats)
    label_array = np.concatenate(labels)
    if limit is not None:
        features, label_array = features[:limit], label_array[:limit]
    print(f"loaded {features.shape[0]:,} positions from {len(feats)} shard(s)")
    return features, label_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NNUE evaluation.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--acc", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 1))

    features, labels = load_data(args.limit)
    n = features.shape[0]
    split = int(n * (1 - args.val_frac))
    order = np.random.default_rng(0).permutation(n)
    features, labels = features[order], labels[order]

    train_ds = ShardDataset(features[:split], labels[:split])
    val_ds = ShardDataset(features[split:], labels[split:])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = NNUE(args.acc, args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]
        sched.step()

        model.eval()
        val_loss = 0.0
        val_seen = 0
        abs_cp = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                val_loss += float(loss_fn(pred, yb)) * xb.shape[0]
                abs_cp += float(torch.abs(pred - yb).sum()) * CP_SCALE
                val_seen += xb.shape[0]
        train_mse = running / max(seen, 1)
        val_mse = val_loss / max(val_seen, 1)
        mae_cp = abs_cp / max(val_seen, 1)
        print(
            f"epoch {epoch + 1:2d}/{args.epochs}  train_mse={train_mse:.4f}  "
            f"val_mse={val_mse:.4f}  val_MAE={mae_cp:6.1f} cp  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {"state": model.state_dict(), "acc": args.acc, "hidden": args.hidden},
                OUT_DIR / "net.pt",
            )

    print(f"best val_mse={best_val:.4f}; saved {OUT_DIR / 'net.pt'}")


if __name__ == "__main__":
    main()
