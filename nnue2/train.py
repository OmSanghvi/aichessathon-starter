"""Train the NNUE evaluation.

Architecture: NUM_FEATURES -> ACC -> 32 -> 1, clipped ReLU throughout, small on
purpose. Inference speed decides whether a net is worth using at all: measured, an
accumulator of 256 costs about 15% of the search, and the net has to be accurate
enough to more than pay that back.

Three things here are quantisation constraints rather than accuracy choices:
  * clipped ReLU (clamp to [0, 1]) matches the integer kernel's saturation, so the
    quantised net behaves like the trained one instead of drifting at the extremes
  * the output stays LINEAR in centipawns, because sunfish's pruning constants
    (QS, QS_A, LMR, NULL_MARGIN, the calmness test) are all expressed in those
    units and the net has to speak the same language as the table it replaces
  * weights are kept small enough to survive int16 scaling

THE LOSS is the main change from v1. v1 regressed raw centipawns and plateaued at
93 cp validation error, which lost games badly. Here the error is measured after
passing both sides through a sigmoid:

    loss = MSE( sigmoid(pred/K), sigmoid(target/K) )

That is standard NNUE practice. A linear centipawn loss spends most of its gradient
on already-decided positions, where being wrong by 200 cp changes no decision. In
sigmoid space those positions saturate and the gradient concentrates on the
balanced middle, which is where evaluation actually picks moves. The output remains
linear centipawns, so quantisation is unaffected.

Resumable, checkpoints every epoch, early stopping on validation.

    .venv/bin/python nnue2/train.py --epochs 60
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nnue2.features import NUM_FEATURES, codes_to_features  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "out"

# Centipawns per sigmoid unit. 400 is the usual choice: it puts a one-pawn edge
# well inside the responsive part of the curve.
K_CP = 400.0


class Positions(Dataset):
    """Expands stored boards into sparse feature indices on demand.

    Storing boards rather than dense 3072-wide rows keeps the dataset small and,
    more importantly, lets the feature set change without regenerating data.
    """

    def __init__(self, boards: np.ndarray, turns: np.ndarray, labels: np.ndarray) -> None:
        self.boards = boards
        self.turns = turns
        self.labels = labels

    def __len__(self) -> int:
        return self.boards.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = codes_to_features(self.boards[i], int(self.turns[i]))
        dense = torch.zeros(NUM_FEATURES, dtype=torch.float32)
        dense[torch.from_numpy(idx.astype(np.int64))] = 1.0
        return dense, torch.tensor([float(self.labels[i])], dtype=torch.float32)


class Residuals(Positions):
    """Targets become label - piece_square_score, and the anchor is returned too.

    Predicting the residual is what keeps the evaluation compatible with a search
    that prunes on pos.score. Measured, the net and the tables disagreed by 311 cp
    on average while the search cut lines using the tables and scored leaves with
    the net; anchoring removes that contradiction.
    """

    def __init__(self, boards, turns, labels, anchors) -> None:  # type: ignore[no-untyped-def]
        super().__init__(boards, turns, labels)
        self.anchors = anchors

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        dense, _ = super().__getitem__(i)
        residual = float(self.labels[i]) - float(self.anchors[i])
        return dense, torch.tensor([residual], dtype=torch.float32)


class NNUE(nn.Module):
    def __init__(self, acc: int = 256, hidden: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.ft = nn.Linear(NUM_FEATURES, acc)
        self.l1 = nn.Linear(acc, hidden)
        self.l2 = nn.Linear(hidden, 1)
        # v1 memorised: train MSE 0.06 against validation 0.15. Dropout on the
        # narrow layer plus weight decay is the cheap counter.
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(self.ft(x), 0.0, 1.0)
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = self.drop(x)
        return self.l2(x) * K_CP  # output is centipawns


def wdl_loss(pred_cp: torch.Tensor, target_cp: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(
        torch.sigmoid(pred_cp / K_CP), torch.sigmoid(target_cp / K_CP)
    )


def load(data: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards = sorted(data.glob("shard_*.npz"))
    if not shards:
        raise SystemExit("no shards in nnue2/data - run nnue2/gen.py first")
    b, t, y = [], [], []
    total = 0
    for path in shards:
        with np.load(path) as z:
            b.append(z["boards"])
            t.append(z["turns"])
            y.append(z["labels"])
        total += y[-1].shape[0]
        if limit is not None and total >= limit:
            break
    boards = np.concatenate(b)
    turns = np.concatenate(t)
    labels = np.concatenate(y)
    if limit is not None:
        boards, turns, labels = boards[:limit], turns[:limit], labels[:limit]
    print(f"loaded {labels.shape[0]:,} positions from {len(b)} shard(s)")
    if labels.shape[0] < 1_000_000:
        print("WARNING: v1 had 500k and plateaued at 93cp MAE, losing 12.5%. "
              "More data is the single biggest lever.")
    return boards, turns, labels


def main() -> None:
    global OUT

    p = argparse.ArgumentParser(description="Train the NNUE evaluation.")
    p.add_argument(
        "--out", default=str(OUT),
        help="checkpoint directory (use a separate directory for each evaluator experiment)",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--data-dir", default=str(DATA))
    p.add_argument(
        "--hard-data-dir", default="",
        help="optional blunder-mined shards; used only for training, never validation",
    )
    p.add_argument(
        "--hard-repeat", type=int, default=4,
        help="how often to repeat each mined child position in the training split",
    )
    p.add_argument("--acc", type=int, default=256)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.03)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--residual", action="store_true",
        help="predict label - piece_square_score instead of the label; keeps the "
             "evaluation anchored so sunfish's pruning margins stay valid",
    )
    p.add_argument(
        "--residual-anchor", choices=("nengine", "sunfish"), default="nengine",
        help="static score to subtract for residual training (default: current nengine)",
    )
    args = p.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    boards, turns, labels = load(Path(args.data_dir), args.limit)
    n = labels.shape[0]
    order = np.random.default_rng(0).permutation(n)
    boards, turns, labels = boards[order], turns[order], labels[order]
    split = int(n * (1 - args.val_frac))

    train_boards = boards[:split]
    train_turns = turns[:split]
    train_labels = labels[:split]
    if args.hard_data_dir:
        if args.hard_repeat < 1:
            raise SystemExit("--hard-repeat must be positive")
        hard_boards, hard_turns, hard_labels = load(Path(args.hard_data_dir), None)
        train_boards = np.concatenate((
            train_boards,
            np.tile(hard_boards, (args.hard_repeat, 1)),
        ))
        train_turns = np.concatenate((train_turns, np.tile(hard_turns, args.hard_repeat)))
        train_labels = np.concatenate((train_labels, np.tile(hard_labels, args.hard_repeat)))
        print(f"added {hard_labels.shape[0]:,} mined rows x{args.hard_repeat} to training only")

    if args.residual:
        if args.residual_anchor == "nengine":
            from nnue2.nengine_pst import scores_batch
        else:
            from nnue2.pst import pst_scores_batch as scores_batch

        print(f"residual mode: target is label - {args.residual_anchor} static score")
        t0 = time.time()
        anchors = scores_batch(boards, turns)
        res = labels.astype(np.int32) - anchors
        print(f"  computed {n:,} anchors in {time.time() - t0:.0f}s")
        print(f"  residual: mean {res.mean():+.0f}  std {res.std():.0f}  "
              f"vs raw label std {labels.std():.0f}")
        train_ds: Dataset = Residuals(
            train_boards, train_turns, train_labels,
            scores_batch(train_boards, train_turns))
        val_ds: Dataset = Residuals(
            boards[split:], turns[split:], labels[split:], anchors[split:])
    else:
        train_ds = Positions(train_boards, train_turns, train_labels)
        val_ds = Positions(boards[split:], turns[split:], labels[split:])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

    model = NNUE(args.acc, args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    best = float("inf")
    ckpt_path = OUT / "net.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        start_epoch = ck.get("epoch", 0)
        best = ck.get("best", float("inf"))
        print(f"resumed from epoch {start_epoch}, best val {best:.5f}")

    stale = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run = seen = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = wdl_loss(model(xb), yb)
            loss.backward()
            opt.step()
            run += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]
        sched.step()

        model.eval()
        vl = vseen = mae = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                vl += float(wdl_loss(pred, yb)) * xb.shape[0]
                mae += float(torch.abs(pred - yb).sum())
                vseen += xb.shape[0]
        val = vl / max(vseen, 1)
        mae_cp = mae / max(vseen, 1)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train {run / max(seen, 1):.5f}  "
              f"val {val:.5f}  val_MAE {mae_cp:6.1f} cp  "
              f"lr {sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.0f}s)", flush=True)

        if val < best:
            best, stale = val, 0
            torch.save({"state": model.state_dict(), "acc": args.acc,
                        "hidden": args.hidden, "epoch": epoch + 1, "best": best,
                        "val_mae_cp": mae_cp, "residual": args.residual,
                        "residual_anchor": args.residual_anchor},
                       ckpt_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no improvement in {args.patience} epochs")
                break

    print(f"\nbest val {best:.5f}; saved {ckpt_path}")
    print("Guide: v1 plateaued near 93 cp MAE and lost 12.5% in games. "
          "Under ~50 cp is where this becomes worth shipping.")


if __name__ == "__main__":
    main()
