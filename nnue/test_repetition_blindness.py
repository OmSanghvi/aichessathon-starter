"""Confirm (or refute) that the engine is blind to repetition draws.

HYPOTHESIS
    The engine cannot see a repetition cycle it discovers inside its own search
    tree. So when its king is being perpetually checked it evaluates each escape
    as "+200, fine" and never realises the sequence is a FORCED DRAW. That would
    explain both facts that killed the other theories: extra time does not help
    (the draw is invisible at any depth) and king-table tuning does not help (the
    problem is not which square the king prefers).

TWO INDEPENDENT TESTS, neither of which modifies the search.

  TEST 1 - score disagreement.
    Take positions a reference engine scores as dead level (0.00) BECAUSE of
    perpetual check. Ask our engine for its own score. If ours reports a large
    advantage where the truth is 0.00, it cannot see the draw. That is the
    smoking gun, and it needs no code change to observe.

  TEST 2 - supply the knowledge and see if the move changes.
    The agent already scores a position as a draw when it appears in `_history`
    (the positions it has been asked about this game). So we can FEED it the game
    history and see whether its move choice improves. If knowing about the
    repetition fixes the move, the missing knowledge was the cause. If the move
    does not change, the hypothesis is wrong and the search is not the problem.

A confirmed hypothesis looks like: big score disagreement in TEST 1, and better
moves in TEST 2. Anything else means do NOT touch the search.
"""

import importlib
import io
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def game():
    g = chess.pgn.read_game(io.StringIO(f'[FEN "{START}"]\n[SetUp "1"]\n\n{MOVES}'))
    assert g is not None
    return g


def position_at(fullmove: int, side=chess.BLACK) -> chess.Board:
    board = game().board()
    for mv in game().mainline_moves():
        if board.fullmove_number == fullmove and board.turn == side:
            break
        board.push(mv)
    return board


def our_score(agent, fen: str, think_s: float) -> tuple[int, int]:
    """Our engine's own converged score and depth for a position."""
    pos = agent.board_to_sunfish(chess.Board(fen))
    s = agent.Searcher()
    s.soft = s.deadline = time.time() + think_s
    score = 0
    depth = 0
    try:
        for d, gamma, sc, _m in s.search([pos]):
            if sc >= gamma:
                score, depth = sc, d
    except agent.Stop:
        pass
    return score, depth


def main() -> None:
    import agent

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "Hash": 256})
    limit = chess.engine.Limit(depth=20)

    # ---------------- TEST 1 -------------------------------------------------
    print("=" * 74)
    print("TEST 1: our score vs truth, in positions drawn BY perpetual check")
    print("=" * 74)
    print(f"{'move':>5}{'our score':>12}{'our depth':>11}{'reference':>12}{'gap':>9}")
    gaps = []
    for fullmove in (45, 46, 48, 50):
        board = position_at(fullmove)
        ref = engine.analyse(board, limit)["score"].pov(board.turn).score(mate_score=10000)
        ours, depth = our_score(agent, board.fen(), 3.0)
        gaps.append(abs(ours - ref))
        print(f"{fullmove:>5}{ours:>12}{depth:>11}{ref:>12}{ours - ref:>+9}")
    print(f"\nmean absolute gap: {sum(gaps) / len(gaps):.0f} cp")
    print("A large gap here = our engine does not see the forced draw.")

    # ---------------- TEST 2 -------------------------------------------------
    print("\n" + "=" * 74)
    print("TEST 2: does supplying the game history change the move?")
    print("=" * 74)
    print("Positions 39/40/41 threw away 268/213/157 cp with king moves.\n")
    print(f"{'move':>5}{'no history':>13}{'cp':>8}{'with history':>15}{'cp':>8}{'ref':>8}{'cp':>8}")

    total_without = total_with = 0
    for fullmove in (39, 40, 41):
        board = position_at(fullmove)
        fen = board.fen()
        ref = engine.analyse(board, limit)
        ref_san = board.san(ref["pv"][0])
        rb = chess.Board(fen)
        rb.push(ref["pv"][0])
        ref_cp = engine.analyse(rb, limit)["score"].pov(chess.BLACK).score(mate_score=10000)

        results = {}
        for mode in ("empty", "populated"):
            importlib.reload(agent)  # fresh module state, so _history starts empty
            if mode == "populated":
                # Replay OUR side's earlier positions so _history fills exactly as
                # it would in a real game, then ask for this move.
                b = game().board()
                for mv in game().mainline_moves():
                    if b.fullmove_number == fullmove and b.turn == chess.BLACK:
                        break
                    if b.turn == chess.BLACK:
                        agent.get_move(b.fen(), 40_000)
                    b.push(mv)
            uci = agent.get_move(fen, 40_000)
            bb = chess.Board(fen)
            try:
                m = chess.Move.from_uci(uci)
                san = bb.san(m) if m in bb.legal_moves else "ILL"
                bb.push(m)
                cp = engine.analyse(bb, limit)["score"].pov(chess.BLACK).score(
                    mate_score=10000
                )
            except ValueError:
                san, cp = "ILL", -10000
            results[mode] = (san, cp)

        total_without += results["empty"][1]
        total_with += results["populated"][1]
        print(
            f"{fullmove:>5}{results['empty'][0]:>13}{results['empty'][1]:>8}"
            f"{results['populated'][0]:>15}{results['populated'][1]:>8}"
            f"{ref_san:>8}{ref_cp:>8}"
        )

    print(f"\ntotal cp: no history {total_without}, with history {total_with} "
          f"({total_with - total_without:+d})")
    print("\nVERDICT GUIDE")
    print("  large TEST 1 gap AND TEST 2 improves -> hypothesis CONFIRMED, fix the search")
    print("  large TEST 1 gap but TEST 2 flat     -> blind to draws, but history is not")
    print("                                          the lever; needs in-tree detection")
    print("  small TEST 1 gap                     -> hypothesis WRONG, leave search alone")
    engine.quit()


if __name__ == "__main__":
    main()
