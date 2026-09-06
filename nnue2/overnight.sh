#!/bin/bash
# Unattended overnight run: generate, verify, train, quantise.
#
# Nothing here touches the submission. Weights only reach the agent if you
# explicitly copy them to weights/ afterwards, so this is always safe to leave.
#
#   nohup bash nnue2/overnight.sh > nnue2/overnight.log 2>&1 &
#   tail -f nnue2/overnight.log

set -u  # undefined variables are errors; we handle command failures explicitly

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python

GEN_HOURS="${GEN_HOURS:-8}"
EPOCHS="${EPOCHS:-60}"
ACC="${ACC:-256}"

echo "=============================================="
echo "NNUE v2 overnight run: $(date)"
echo "  generation: ${GEN_HOURS}h   epochs: ${EPOCHS}   accumulator: ${ACC}"
echo "=============================================="

# The encoding contract comes first. A mismatch between the training and agent
# feature paths trains a net that plays like noise while looking healthy, which is
# exactly what happened in v1, so it is checked before any compute is spent.
echo
echo "--- [1/5] verifying feature encodings ---"
if ! $PY nnue2/verify_features.py; then
    echo "ABORT: feature encodings disagree. Fix nnue2/features.py before training."
    exit 1
fi

echo
echo "--- [2/5] generating data (${GEN_HOURS}h, resumable) ---"
$PY nnue2/gen.py --hours "${GEN_HOURS}" || echo "generation ended early; continuing with what exists"

echo
echo "--- [3/5] inspecting the dataset ---"
$PY nnue2/inspect_data.py || { echo "ABORT: no usable data"; exit 1; }

echo
echo "--- [4/5] training (${EPOCHS} epochs, resumable) ---"
$PY nnue2/train.py --epochs "${EPOCHS}" --acc "${ACC}" --resume || {
    echo "ABORT: training failed"; exit 1; }

echo
echo "--- [5/5] quantising and verifying the integer path ---"
$PY nnue2/export.py || { echo "ABORT: export failed"; exit 1; }

echo
echo "=============================================="
echo "done: $(date)"
echo
echo "The net is NOT in the submission yet, on purpose. Before shipping:"
echo "  1. check the reported val_MAE. v1 plateaued near 93 cp and lost 12.5% in"
echo "     games; under ~50 cp is where this becomes worth trying."
echo "  2. confirm quantisation error is small (export.py prints it)."
echo "  3. add the agent-side kernel, then play it against baselines/shipped."
echo "     A net has to beat the tables by MORE than the ~15% of search it costs."
echo "=============================================="
