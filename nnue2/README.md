# NNUE v2

A second attempt at a learned evaluation, rebuilt around what went wrong the first
time. Read this before running anything; the changes are deliberate.

## What failed in v1, and what changed

| v1 problem | v2 fix |
| --- | --- |
| 500k positions; validation error plateaued at 93 cp and the net scored 12.5% in games | generation runs unattended for hours and is resumable; target millions |
| data stored FEATURE INDICES, so changing the feature set threw all the data away | data stores the raw board, so features can be redesigned without regenerating |
| plain 768 features (piece x square) carried no king-safety information | king-bucketed features: the net sees piece placement *relative to where our king is* |
| linear centipawn target wasted capacity on already-decided positions | WDL-sigmoid loss, which concentrates gradient on the decisive range |
| train MSE 0.06 vs val 0.15 - memorising | weight decay, dropout on the hidden layer, early stopping on validation |
| feature mismatch between training and the agent nearly shipped silently | one shared module plus a verifier that asserts both paths agree exactly |

The king-bucket change matters most for this engine specifically. The measured
weakness is king safety: in a won position (+657) the engine walked its king
h7-g8-f8-e7-e8-f8-f7 into a perpetual check and drew, and it plays those same king
moves at *any* time budget, so it is missing knowledge rather than search depth.
Plain piece-square features cannot express "this placement is dangerous **because**
my king is here". King buckets can.

## Order of operations

```bash
# 1. Generate data. Resumable - stop and restart freely, it counts existing shards.
#    ~250 positions/sec, so about an hour per million.
.venv/bin/python nnue2/gen.py --hours 8

# 2. Check what you got before training on it.
.venv/bin/python nnue2/inspect_data.py

# 3. Hard validation (feature contract, sign convention, integer path).
#    .venv/bin/python nnue2/validate.py
#
# 3b. Confirm the agent and training feature encodings are identical. NEVER SKIP.
#    A mismatch here trains a net that plays like noise, and nothing looks broken.
.venv/bin/python nnue2/verify_features.py

# 4. Train. Resumable via --resume; checkpoints every epoch.
.venv/bin/python nnue2/train.py --epochs 60

# 5. Quantise to integers and verify the int path matches the float net.
.venv/bin/python nnue2/export.py

# 6. Speed check. If an evaluation costs more than ~8us it is not worth the
#    depth it takes away, and you should shrink the accumulator.
.venv/bin/python nnue2/bench.py
```

## Overnight, in one command

```bash
nohup .venv/bin/bash nnue2/overnight.sh > nnue2/overnight.log 2>&1 &
tail -f nnue2/overnight.log
```

## The honest risk

v1 was slower *and* weaker than the hand-tuned tables. A net only wins if it is
accurate enough to pay for the nodes it costs: measured, an accumulator of 256
costs about 15% of the search. So the bar is not "the net works", it is "the net
beats piece-square tables by more than 15% of search depth is worth". Validation
MAE is the early signal - v1 plateaued near 93 cp and lost badly. Under about
50 cp is where this starts being interesting.

Nothing here shows up in the submission unless `export.py` is run AND the weights
are copied to `weights/`, so training is always safe to leave running.
