"""How wide should the residual clamp be?

Anchoring the net to pos.score took the agent from 13.8% to 46.9%, because the
search prunes on pos.score and an unanchored evaluation made those decisions
incoherent. The clamp is the width of that anchor, and it trades two things off:

  too tight  the net cannot express what it knows; evaluation collapses to the
             piece-square tables and the net is dead weight that costs 11% of a node
  too loose  the evaluation drifts far from pos.score again and the tuned pruning
             margins stop being meaningful, which is the 13.8% failure

The measured p90 divergence at clamp 300 was exactly 300, meaning the clamp binds
often, so this is a live parameter rather than a formality.

Reports, for each candidate clamp, how well the EFFECTIVE evaluation tracks a
reference engine and how far it strays from the tables. The best clamp maximises
accuracy while keeping divergence in the range where pruning still works.
"""

import random
import sys
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402

ENGINE = "/opt/homebrew/bin/stockfish"
CLAMPS = (0, 100, 200, 300, 500, 800, 1200, 10_000)
N = 300


def sample(n: int) -> list[chess.Board]:
    rng = random.Random(11)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(8, 60)):
            mv = list(b.legal_moves)
            if not mv:
                break
            b.push(rng.choice(mv))
            if b.is_game_over():
                break
        if not b.is_game_over() and len(b.piece_map()) >= 10:
            out.append(b)
    return out


def main() -> None:
    if agent._NNUE is None or not agent._NNUE.residual:
        raise SystemExit("needs a residual net loaded")

    eng = chess.engine.SimpleEngine.popen_uci(ENGINE)
    eng.configure({"Threads": 1, "Hash": 256})
    limit = chess.engine.Limit(depth=14)

    rows = []
    for b in sample(N):
        cp = eng.analyse(b, limit)["score"].pov(b.turn).score(mate_score=2000)
        if cp is None:
            continue
        pos = agent.board_to_sunfish(b)
        rows.append((max(-1500, min(1500, cp)), pos.score, agent._NNUE.evaluate(pos.board)))
    eng.quit()

    truth = np.array([r[0] for r in rows], dtype=float)
    pst = np.array([r[1] for r in rows], dtype=float)
    raw = np.array([r[2] for r in rows], dtype=float)

    print(f"{len(rows)} positions\n")
    print(f"{'clamp':>8}{'corr vs truth':>15}{'MAE':>8}{'sign':>7}"
          f"{'mean |eval-pst|':>18}{'p90':>7}{'clamp hits':>12}")
    baseline_corr = float(np.corrcoef(pst, truth)[0, 1])
    for c in CLAMPS:
        eff = pst + np.clip(raw, -c, c)
        corr = float(np.corrcoef(eff, truth)[0, 1])
        mae = float(np.mean(np.abs(eff - truth)))
        sign = float(np.mean(np.sign(eff) == np.sign(truth)))
        div = np.abs(eff - pst)
        hits = float(np.mean(np.abs(raw) > c)) if c < 10_000 else 0.0
        star = "  <-- current" if c == agent.NNUE_MAX_CORRECTION else ""
        print(f"{c:>8}{corr:>15.3f}{mae:>8.0f}{sign:>6.0%}"
              f"{np.mean(div):>18.0f}{np.percentile(div, 90):>7.0f}{hits:>11.0%}{star}")
    print(f"\npiece-square tables alone: corr {baseline_corr:.3f}, "
          f"MAE {np.mean(np.abs(pst - truth)):.0f}")
    print("\nclamp 0 is the tables exactly; 10000 is unanchored (the 13.8% setup).")
    print("Pick the clamp that maximises correlation while keeping mean divergence")
    print("small enough that pos.score-based pruning still means something.")


if __name__ == "__main__":
    main()
