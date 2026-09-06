"""Why did a winning position turn into a threefold repetition?

Two claims to test:
  1. "the bot does not know how to move the king"
  2. "it is not looking 3-4 moves ahead"

Method: reconstruct the game, ask a reference engine whether White was actually
winning during the checking sequence, then measure what depth OUR engine reaches
with the budget it really had at that point in the game, versus with more time.

If our engine finds the win given time but not given its real budget, the problem
is time and depth. If it never finds it, the problem is evaluation or repetition
handling.
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

# White's real clock at these move numbers, from the PGN.
WHITE_CLOCK = {31: 28.3, 35: 20.0, 40: 13.1, 46: 10.1, 48: 10.1, 50: 8.1}


def build(upto_fullmove: int, white_to_move: bool = True) -> chess.Board:
    pgn = f'[FEN "{START}"]\n[SetUp "1"]\n\n{MOVES}'
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    board = game.board()
    for mv in game.mainline_moves():
        if board.fullmove_number == upto_fullmove and board.turn == (
            chess.WHITE if white_to_move else chess.BLACK
        ):
            break
        board.push(mv)
    return board


def main() -> None:
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "Hash": 256})
    deep = chess.engine.Limit(depth=22)

    print("Was White actually winning during the checking sequence?\n")
    print(f"{'move':>5}{'material':>28}{'ref eval':>10}  ref best line")
    for mv in (31, 35, 40, 46, 48, 50):
        board = build(mv)
        info = engine.analyse(board, deep)
        score = info["score"].pov(chess.WHITE)
        pv = info.get("pv", [])[:6]
        line = board.variation_san(pv) if pv else "?"
        wm = sum(
            len(board.pieces(p, chess.WHITE)) * v
            for p, v in ((chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                         (chess.ROOK, 5), (chess.QUEEN, 9))
        )
        bm = sum(
            len(board.pieces(p, chess.BLACK)) * v
            for p, v in ((chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                         (chess.ROOK, 5), (chess.QUEEN, 9))
        )
        print(f"{mv:>5}{f'W {wm} vs B {bm}':>28}{str(score):>10}  {line}")

    print("\n\nDepth our engine reaches, real budget vs generous budget")
    print("(clock_s is what White actually had at that move)\n")
    print(f"{'move':>5}{'clock_s':>9}{'our move @real':>16}{'@30s':>10}{'@120s':>10}"
          f"{'ref best':>12}")
    for mv, clock_s in WHITE_CLOCK.items():
        board = build(mv)
        ref = engine.analyse(board, deep)
        ref_best = board.san(ref["pv"][0]) if ref.get("pv") else "?"

        picks = []
        for budget_s in (clock_s, 30.0, 120.0):
            uci = agent.get_move(board.fen(), int(budget_s * 1000))
            try:
                m = chess.Move.from_uci(uci)
                picks.append(board.san(m) if m in board.legal_moves else "ILL")
            except ValueError:
                picks.append("ILL")
        print(f"{mv:>5}{clock_s:>9.1f}{picks[0]:>16}{picks[1]:>10}{picks[2]:>10}"
              f"{ref_best:>12}")

    engine.quit()


if __name__ == "__main__":
    main()
