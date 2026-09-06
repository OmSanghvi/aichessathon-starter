#!/bin/bash
# Reproducible offline NNUE-v2 run for the current mailbox/Nengine evaluator.
#
# The generated model is intentionally kept under nnue2/out_nengine/. It does
# not reach agent.py or submission.zip without a separate, benchmarked port.
#
# Optional environment overrides:
#   GEN_HOURS=8 EPOCHS=60 DATA_DIR=nnue2/data bash nnue2/train_nengine.sh

set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
DATA_DIR="${DATA_DIR:-nnue2/data}"
GEN_HOURS="${GEN_HOURS:-8}"
EPOCHS="${EPOCHS:-60}"
CASTLE_OPENING_SHARE="${CASTLE_OPENING_SHARE:-0.75}"
# ``out_nengine`` predates the PeSTO/mobility evaluator and was trained against
# a stale residual anchor. ``out_nengine_pesto`` uses only four king regions and
# plateaued at 79 cp MAE. Never resume either: this v3 model has eight regions.
OUT="${OUT:-nnue2/out_nengine_k8}"

echo "[1/5] current evaluator anchor"
"$PY" nnue2/verify_nengine_anchor.py

echo "[2/5] extend raw-board data with current Nengine self-play"
"$PY" nnue2/gen.py --hours "$GEN_HOURS" --data-dir "$DATA_DIR" \
  --castle-opening-share "$CASTLE_OPENING_SHARE"

echo "[3/5] inspect data before training"
"$PY" nnue2/inspect_data.py --data-dir "$DATA_DIR"

echo "[4/5] train a residual relative to the current Nengine evaluator"
"$PY" nnue2/train.py --data-dir "$DATA_DIR" --out "$OUT" --epochs "$EPOCHS" \
  --acc 256 --hidden 64 --resume --residual --residual-anchor nengine

echo "[5/5] export integer candidate (offline only)"
"$PY" nnue2/export.py --checkpoint "$OUT/net.pt" --out "$OUT/nnue_weights.npz"

echo "Candidate ready at $OUT/nnue_weights.npz. Benchmark before shipping."
