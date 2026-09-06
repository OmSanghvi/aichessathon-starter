"""Does the jitted search work, and how much depth does the speed actually buy?

The pure-Python engine reaches depth 7-9 in a middlegame at about 55,000 nodes a
second. Perft on the new generator runs at 15-22M. This measures what the real
search - with evaluation, ordering, a transposition table and quiescence - achieves,
and confirms it finds the tactics the old engine found.
"""

import sys
import time
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nengine.board import IS_SLIDER, N_OFFSETS, OFFSETS, mv_from, mv_promo, mv_to  # noqa: E402
from nengine.search import (  # noqa: E402
    MAX_PLY,
    PST,
    PST_KING_END,
    PST_KING_MID,
    ZOB_CASTLE,
    ZOB_EP,
    ZOB_PIECE,
    ZOB_SIDE,
    new_tt,
    search_root,
    zobrist,
)
from nengine.test_perft import to_arrays  # noqa: E402

SQ64_NAME = {}
for _s in range(64):
    SQ64_NAME[91 + (_s & 7) - 10 * (_s >> 3)] = chess.square_name(_s)

CODE_TO_PROMO = {2: "n", 3: "b", 4: "r", 5: "q", 8: "n", 9: "b", 10: "r", 11: "q"}


def move_to_uci(m: int) -> str:
    u = SQ64_NAME[mv_from(m)] + SQ64_NAME[mv_to(m)]
    p = mv_promo(m)
    return u + CODE_TO_PROMO.get(p, "") if p else u


def run(fen: str, max_depth: int, node_limit: int = 200_000_000):
    board, side, cr, ep = to_arrays(chess.Board(fen))
    tt = new_tt()
    killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
    history = np.zeros((2, 120, 120), dtype=np.int32)
    game_hashes = np.zeros(600, dtype=np.int64)
    best = 0
    rows = []
    for d in range(1, max_depth + 1):
        counters = np.zeros(2, dtype=np.int64)
        root_hash = zobrist(board, side, cr, ep, ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP)
        t0 = time.perf_counter()
        score, mv = search_root(
            board, side, cr, ep, d,
            OFFSETS, N_OFFSETS, IS_SLIDER, PST, PST_KING_MID, PST_KING_END,
            tt[0], tt[1], tt[2], tt[3], tt[4],
            killers, history, counters, node_limit,
            ZOB_PIECE, ZOB_SIDE, ZOB_CASTLE, ZOB_EP, best, game_hashes, 1, root_hash,
        )
        dt = time.perf_counter() - t0
        if counters[1] == 1:
            rows.append((d, None, None, counters[0], dt, True))
            break
        best = mv
        rows.append((d, score, move_to_uci(mv), counters[0], dt, False))
    return rows, best


def main() -> None:
    print("compiling...")
    t0 = time.time()
    run(chess.STARTING_FEN, 2)
    print(f"  compiled in {time.time() - t0:.1f}s (must fit the 60s import budget)\n")

    positions = [
        ("startpos", chess.STARTING_FEN, 8),
        ("middlegame", "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8", 8),
        ("complex mid", "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7", 8),
        ("the lost endgame", "6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42", 10),
    ]
    for name, fen, md in positions:
        print(f"=== {name} ===")
        rows, _ = run(fen, md)
        for d, score, mv, nodes, dt, aborted in rows:
            if aborted:
                print(f"  depth {d}: aborted")
                continue
            nps = nodes / dt if dt > 0 else 0
            print(f"  depth {d:>2}: score {score:>6}  best {mv:<6} "
                  f"{nodes:>10,} nodes  {dt:>6.2f}s  {nps:>10,.0f} nps")
        print()

    print("=== tactics: does it find them? ===")
    tests = [
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8", "back-rank mate"),
        ("k7/8/1K6/8/8/8/8/7R w - - 0 1", "h1h8", "KRK mate"),
        ("7k/R7/1R6/8/8/8/8/6K1 w - - 0 1", "b6b8", "ladder mate"),
        ("8/1P6/8/8/8/2k5/8/6K1 w - - 0 1", "b7b8q", "promotion"),
        ("rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 1", "f3g4", "win queen"),
        ("r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", "f6a6", "mate in 2"),
        ("6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42", "g6e4", "the Be4 position"),
    ]
    ok = 0
    for fen, want, label in tests:
        rows, best = run(fen, 8)
        got = move_to_uci(best)
        good = got == want
        ok += good
        print(f"  {'PASS' if good else 'note'} {label:<18} got {got:<6} want {want}")
    print(f"  {ok}/{len(tests)}")


if __name__ == "__main__":
    main()
