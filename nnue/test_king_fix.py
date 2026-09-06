"""Did the king-table phase fix change the three losing king moves?

Moves 39, 40 and 41 each threw away 150-270cp by putting the king on the wrong
square, and the reference engine wanted a central square every time. This checks
what our engine plays in those exact positions now, and scores the result.

Compares the current agent against baselines/fast_pst, which still has the
queens-present phase test, so the difference isolates the change.
"""

import importlib.util
import io
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent as new_agent  # noqa: E402

ENGINE = "/opt/homebrew/bin/stockfish"
START = "r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQR1K1 b kq - 0 8"
MOVES = """8... Bb7 9. Ng5 O-O 10. c3 h6 11. Bxf7+ Rxf7 12. Nxf7 Kxf7 13. Qb3+ d5
14. a4 Na5 15. Qc2 dxe4 16. dxe4 Bxe4 17. Rxe4 Nxe4 18. axb5 axb5 19. Nd2 Nd6
20. b4 Nac4 21. Rxa8 Qxa8 22. Nxc4 bxc4 23. h3 Bf6 24. Bb2 Qd5 25. Qe2 Qd3
26. Qe1 e4 27. Qc1 Nb5 28. Kh2 c6 29. Qf4 Qc2 30. Bc1 Nxc3 31. Qc7+ Be7
32. Qf4+ Kg8 33. Qb8+ Kh7 34. Qc7 Bxb4 35. Qf4 Nd1 36. Qf5+ Kg8 37. Qe6+ Kf8
38. Qc8+ Ke7 39. Qb7+ Ke8 40. Qxc6+ Kf8 41. Qc8+ Kf7 42. Qf5+ Kg8"""

TARGETS = {39: 64.0, 40: 41.7, 41: 39.1}


def load_old() -> object:
    path = ROOT / "baselines" / "fast_pst" / "agent.py"
    spec = importlib.util.spec_from_file_location("fast_pst_agent", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["fast_pst_agent"] = m
    spec.loader.exec_module(m)
    return m


def position_at(fullmove: int) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(f'[FEN "{START}"]\n[SetUp "1"]\n\n{MOVES}'))
    assert game is not None
    board = game.board()
    for mv in game.mainline_moves():
        if board.fullmove_number == fullmove and board.turn == chess.BLACK:
            break
        board.push(mv)
    return board


def main() -> None:
    old = load_old()
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "Hash": 256})
    limit = chess.engine.Limit(depth=20)

    print(f"{'move':>5}{'old':>10}{'old cp':>9}{'new':>10}{'new cp':>9}{'ref':>8}{'ref cp':>9}")
    old_total = new_total = 0
    for fullmove, clock_s in TARGETS.items():
        board = position_at(fullmove)
        fen = board.fen()
        ref = engine.analyse(board, limit)
        ref_best = board.san(ref["pv"][0])
        b = chess.Board(fen)
        b.push(ref["pv"][0])
        ref_cp = engine.analyse(b, limit)["score"].pov(chess.BLACK).score(mate_score=10000)

        row = {}
        for label, mod in (("old", old), ("new", new_agent)):
            uci = mod.get_move(fen, int(clock_s * 1000))  # type: ignore[attr-defined]
            bb = chess.Board(fen)
            try:
                m = chess.Move.from_uci(uci)
                san = bb.san(m) if m in bb.legal_moves else "ILL"
                bb.push(m)
                cp = engine.analyse(bb, limit)["score"].pov(chess.BLACK).score(mate_score=10000)
            except ValueError:
                san, cp = "ILL", -10000
            row[label] = (san, cp)
        old_total += row["old"][1]
        new_total += row["new"][1]
        print(
            f"{fullmove:>5}{row['old'][0]:>10}{row['old'][1]:>9}"
            f"{row['new'][0]:>10}{row['new'][1]:>9}{ref_best:>8}{ref_cp:>9}"
        )

    print(f"\ntotal cp across the three king moves: old {old_total}, new {new_total}")
    print(f"change: {new_total - old_total:+d} cp")
    engine.quit()


if __name__ == "__main__":
    main()
