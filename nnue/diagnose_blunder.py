"""Is the collapse a SEARCH problem or an EVALUATION problem?

The distinction decides everything about where to spend effort:

  * if more thinking time fixes the move, the engine's judgement is fine and it
    simply did not search deep enough -> spend effort on speed and pruning.
  * if it plays the same bad move no matter how long it thinks, it MIS-JUDGES the
    position -> spend effort on the evaluation.

We replay the losing game to the worst blunder, then ask our engine for a move at
increasing budgets and compare against a reference engine's verdict.
"""

import io
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

ENGINE = "/opt/homebrew/bin/stockfish"

START = "rnbqk1nr/bp3ppp/p7/3p4/P7/1N6/1PP2PPP/R1BQKBNR w KQkq - 2 8"
MOVES = """8. Be3 Bxe3 9. Qe2 Ne7 10. Qxe3 Nc6 11. Nf3 O-O 12. Rd1 Nf5 13. Qc5 Be6
14. Bd3 Qf6 15. c3 Rfc8 16. Qb6 Ne5 17. Be2 Nxf3+ 18. Bxf3 Qe7 19. O-O Rc6
20. Qb4 Qxb4 21. cxb4 Rd8 22. Nc5 b5 23. axb5 axb5 24. Rfe1 Nh4 25. Be2 Rb8
26. Nxe6 fxe6 27. g3 Ng6 28. Bd3 Ne7 29. Kg2 d4 30. Re5 Rcb6 31. Be4 Rd8
32. Kf3 g6 33. Rc5 h6 34. Bd3 Nd5 35. Bxg6 Nxb4 36. Rc7 d3 37. Ke4 d2
38. f4 Rdd6 39. Rh7 Rbc6 40. Re7 Rd3 41. Ke5 Re3+"""

# Stop just before the worst blunder (42. Kd4, which threw away 448 cp).
TARGET_FULLMOVE = 42


def build_target() -> chess.Board:
    pgn = f'[FEN "{START}"]\n[SetUp "1"]\n\n{MOVES}'
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    board = game.board()
    for move in game.mainline_moves():
        if board.fullmove_number == TARGET_FULLMOVE and board.turn == chess.WHITE:
            break
        board.push(move)
    return board


def main() -> None:
    board = build_target()
    print(f"position before the blunder (move {TARGET_FULLMOVE}, White to move):")
    print(board)
    print(f"\nfen: {board.fen()}\n")

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "Hash": 128})

    ref = engine.analyse(board, chess.engine.Limit(depth=20))
    best = ref["pv"][0]
    print(f"reference best: {board.san(best)}  eval {ref['score'].pov(chess.WHITE).score()} cp")

    played = chess.Move.from_uci("d4d4") if False else None
    blunder = None
    for m in board.legal_moves:
        if board.san(m) == "Kd4":
            blunder = m
    if blunder is not None:
        board.push(blunder)
        after = engine.analyse(board, chess.engine.Limit(depth=20))
        print(f"what it played: Kd4      eval {after['score'].pov(chess.WHITE).score()} cp")
        board.pop()

    print("\nour engine, increasing time budget:")
    print(f"{'clock ms':>10} {'move':>8} {'ref eval after':>16} {'verdict':>10}")
    for clock in (2_000, 5_000, 15_000, 60_000, 240_000):
        t0 = time.time()
        uci = agent.get_move(board.fen(), clock)
        spent = time.time() - t0
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        board.push(move)
        ev = engine.analyse(board, chess.engine.Limit(depth=18))
        score = ev["score"].pov(chess.WHITE).score(mate_score=10000)
        board.pop()
        verdict = "ok" if score > -150 else "BAD"
        print(f"{clock:>10} {san:>8} {score:>16} {verdict:>10}   ({spent:.1f}s)")

    engine.quit()


if __name__ == "__main__":
    main()
