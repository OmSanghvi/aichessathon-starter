"""Where did BLACK throw away a winning position?

Black was about +585 at move 35 and dead level by move 46, drawn by perpetual
check. This walks Black's moves through that window and reports the evaluation
swing for each, from Black's point of view, to find the move that let the checks
become permanent.

Then, for the worst position, it asks our own engine what it plays at its real
budget versus a generous one - which separates "could not see far enough" from
"does not understand king safety".
"""

import io
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

ENGINE = "/opt/homebrew/bin/stockfish"

START = "r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQR1K1 b kq - 0 8"
MOVES = """8... Bb7 9. Ng5 O-O 10. c3 h6 11. Bxf7+ Rxf7 12. Nxf7 Kxf7 13. Qb3+ d5
14. a4 Na5 15. Qc2 dxe4 16. dxe4 Bxe4 17. Rxe4 Nxe4 18. axb5 axb5 19. Nd2 Nd6
20. b4 Nac4 21. Rxa8 Qxa8 22. Nxc4 bxc4 23. h3 Bf6 24. Bb2 Qd5 25. Qe2 Qd3
26. Qe1 e4 27. Qc1 Nb5 28. Kh2 c6 29. Qf4 Qc2 30. Bc1 Nxc3 31. Qc7+ Be7
32. Qf4+ Kg8 33. Qb8+ Kh7 34. Qc7 Bxb4 35. Qf4 Nd1 36. Qf5+ Kg8 37. Qe6+ Kf8
38. Qc8+ Ke7 39. Qb7+ Ke8 40. Qxc6+ Kf8 41. Qc8+ Kf7 42. Qf5+ Kg8 43. Qe6+ Kh7
44. Qf5+ g6 45. Qf7+ Kh8 46. Bf4 Bc3 47. Qf8+ Kh7 48. Qxh6+ Kg8 49. Qxg6+ Kh8
50. Qh6+ Kg8 51. Qg6+ Kh8 52. Qh6+ Kg8"""

# Black's clock at these moves, from the PGN.
BLACK_CLOCK = {31: 64.0, 34: 57.0, 35: 55.1, 40: 41.7, 44: 33.4, 46: 32.3}


def main() -> None:
    game = chess.pgn.read_game(io.StringIO(f'[FEN "{START}"]\n[SetUp "1"]\n\n{MOVES}'))
    assert game is not None
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "Hash": 256})
    limit = chess.engine.Limit(depth=20)

    board = game.board()
    mistakes = []
    worst_fen = None
    print("BLACK's moves, evaluation from BLACK's point of view\n")
    print(f"{'move':>6} {'played':>8} {'before':>8} {'after':>8} {'swing':>7}  better")
    for mv in game.mainline_moves():
        if board.turn != chess.BLACK:
            board.push(mv)
            continue
        info = engine.analyse(board, limit)
        before = info["score"].pov(chess.BLACK).score(mate_score=10000)
        best = board.san(info["pv"][0]) if info.get("pv") else "?"
        played = board.san(mv)
        num = board.fullmove_number
        fen = board.fen()
        board.push(mv)
        after = engine.analyse(board, limit)["score"].pov(chess.BLACK).score(mate_score=10000)
        swing = before - after
        flag = ""
        if swing >= 120:
            flag = "  <-- THREW AWAY"
            mistakes.append((num, played, before, after, swing, best, fen))
        if num >= 28:
            print(f"{num:>6} {played:>8} {before:>8} {after:>8} {swing:>7}  {best}{flag}")

    print(f"\n{len(mistakes)} Black moves losing 120+ cp, worst first:\n")
    for num, played, before, after, swing, best, fen in sorted(mistakes, key=lambda m: -m[4]):
        print(f"  move {num:>3}: {played:<8} {before:>6} -> {after:>6}"
              f"  (threw away {swing:>4} cp)   better: {best}")
        if worst_fen is None:
            worst_fen = (num, fen, best, played)

    if worst_fen is not None:
        num, fen, best, played = worst_fen
        print(f"\n\nWORST POSITION (Black to move, move {num}):")
        print(chess.Board(fen))
        print(f"fen: {fen}")
        print(f"  played {played}, reference prefers {best}")
        clock = BLACK_CLOCK.get(num, 40.0)
        print(f"\n  our engine, real clock {clock:.0f}s vs generous:")
        for budget_s in (clock, 30.0, 120.0, 300.0):
            uci = agent.get_move(fen, int(budget_s * 1000))
            b = chess.Board(fen)
            try:
                m = chess.Move.from_uci(uci)
                san = b.san(m) if m in b.legal_moves else "ILLEGAL"
            except ValueError:
                san = "ILLEGAL"
            b2 = chess.Board(fen)
            b2.push(chess.Move.from_uci(uci))
            ev = engine.analyse(b2, limit)["score"].pov(chess.BLACK).score(mate_score=10000)
            print(f"    @{budget_s:>5.0f}s -> {san:<8} (leaves Black at {ev:+d} cp)")

    engine.quit()


if __name__ == "__main__":
    main()
