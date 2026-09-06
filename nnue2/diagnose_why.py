"""Why does a more accurate net still lose badly?

v1 (93 cp MAE, plain features, 500k positions) scored 12.5%.
v2 (79 cp MAE, king buckets, 6.6M positions) scored 13.8%.

A large accuracy gain producing no result means accuracy is not the binding
constraint. Two candidate explanations, and they imply opposite fixes:

  A. THE NET IS SIMPLY WORSE THAN THE TABLES it replaces. MAE against a deep
     search is not the same as ranking moves correctly. If the piece-square score
     correlates BETTER with truth than the net does, no integration work helps and
     the whole approach is dead.

  B. THE SEARCH IS INCOHERENT. The net replaces only the leaf evaluation, but
     sunfish still PRUNES using the incremental piece-square score: futility
     ceilings, the null-move margin, the calmness test and move ordering are all
     expressed in pos.score. So the search decides what to cut using one value
     function and scores what survives with a different one. If the two disagree
     substantially, pruning removes lines the net would have liked, and a better
     net cannot repair that. The fix is to anchor the net to pos.score.

This measures both: correlation and rank agreement of each evaluation against a
reference engine, and how far the net and the tables diverge.
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
N_POSITIONS = 260


def sample_positions(n: int) -> list[chess.Board]:
    rng = random.Random(7)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(8, 60)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
            if b.is_game_over():
                break
        if not b.is_game_over() and len(b.piece_map()) >= 10:
            out.append(b)
    return out


def main() -> None:
    if agent._NNUE is None:
        raise SystemExit("net not loaded")

    eng = chess.engine.SimpleEngine.popen_uci(ENGINE)
    eng.configure({"Threads": 1, "Hash": 256})
    limit = chess.engine.Limit(depth=14)

    boards = sample_positions(N_POSITIONS)
    truth, net_cp, pst_cp = [], [], []
    for b in boards:
        cp = eng.analyse(b, limit)["score"].pov(b.turn).score(mate_score=2000)
        if cp is None:
            continue
        pos = agent.board_to_sunfish(b)
        truth.append(max(-1500, min(1500, cp)))
        # Measure the EFFECTIVE evaluation the search actually uses. For a residual
        # net that is pos.score plus the clamped correction; comparing the raw
        # correction against absolute truth would be meaningless.
        raw = agent._NNUE.evaluate(pos.board)
        if agent._NNUE.residual:
            limit_c = agent.NNUE_MAX_CORRECTION
            effective = pos.score + max(-limit_c, min(limit_c, raw))
        else:
            effective = raw
        net_cp.append(effective)
        pst_cp.append(pos.score)

    t = np.array(truth, dtype=float)
    nn = np.array(net_cp, dtype=float)
    ps = np.array(pst_cp, dtype=float)

    print(f"{len(t)} positions, reference at depth {limit.depth}\n")
    print("EXPLANATION A: is the net actually better than the tables?")
    print(f"{'evaluation':<14}{'corr vs truth':>15}{'MAE vs truth':>14}{'sign agree':>12}")
    for name, v in (("net (effective)", nn), ("piece-square", ps)):
        corr = float(np.corrcoef(v, t)[0, 1])
        mae = float(np.mean(np.abs(v - t)))
        sign = float(np.mean(np.sign(v) == np.sign(t)))
        print(f"{name:<14}{corr:>15.3f}{mae:>14.0f}{sign:>11.0%}")

    print("\nEXPLANATION B: do the net and the tables agree with each other?")
    corr_np = float(np.corrcoef(nn, ps)[0, 1])
    diff = nn - ps
    print(f"  corr(net, piece-square) : {corr_np:.3f}")
    print(f"  mean |net - pst|        : {np.mean(np.abs(diff)):.0f} cp")
    print(f"  p90 |net - pst|         : {np.percentile(np.abs(diff), 90):.0f} cp")
    print(f"  sign disagreement       : {np.mean(np.sign(nn) != np.sign(ps)):.0%}")
    print("\n  The search prunes with pos.score but scores leaves with the net.")
    print("  Large divergence here means those two decisions are incoherent.")

    print("\nVERDICT")
    corr_net = float(np.corrcoef(nn, t)[0, 1])
    corr_pst = float(np.corrcoef(ps, t)[0, 1])
    if corr_net < corr_pst:
        print("  The net correlates WORSE with truth than the tables it replaces.")
        print("  -> Explanation A. The net is the problem, not the integration.")
        print("     Anchoring or retuning will not save it; it needs to be a")
        print("     genuinely better evaluation first.")
    elif np.mean(np.abs(diff)) > 150:
        print("  The net is better than the tables, but the two disagree widely,")
        print("  so pruning and scoring are pulling in different directions.")
        print("  -> Explanation B. Anchor the net to pos.score (train on the")
        print("     RESIDUAL) so the tuned pruning margins stay valid.")
    else:
        print("  Neither explanation is clear-cut; look elsewhere.")
    eng.quit()


if __name__ == "__main__":
    main()
