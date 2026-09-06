#!/bin/bash
# Offline high-swing mistake mining for the current Nengine submission.
#
# The result is raw-board training data plus an auditable JSONL list of FENs and
# move losses. It never affects agent.py until a separately benchmarked model is
# trained and enabled.

set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
GAMES="${GAMES:-100}"
THINK_MS="${THINK_MS:-2200}"
DEPTH="${DEPTH:-14}"
THRESHOLD="${THRESHOLD:-120}"
OUT="${OUT:-nnue2/blunders}"

"$PY" nnue2/mine_blunders.py --games "$GAMES" --think-ms "$THINK_MS" \
  --depth "$DEPTH" --threshold "$THRESHOLD" --out "$OUT"
"$PY" nnue2/inspect_data.py --data-dir "$OUT"

echo "Mined corpus: $OUT"
echo "Use it in training with: --hard-data-dir $OUT --hard-repeat 4"
