"""Find where a game was actually lost, and what kind of mistake it was.

Replays a PGN, and for every move by the side under test asks a reference engine
for the evaluation before and after. A large drop is a mistake; the size and the
character of the position tell us whether we are losing games to tactics we did
not see deep enough to find, or to positions we mis-JUDGED.

That distinction decides where effort should go: search depth versus evaluation.
Reference engine is used for analysis only and never ships.
"""

import argparse
import io
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENGINE = "/opt/homebrew/bin/stockfish"

PGN = """
[FEN "rnbqk1nr/bp3ppp/p7/3p4/P7/1N6/1PP2PPP/R1BQKBNR w KQkq - 2 8"]
[SetUp "1"]

8. Be3 Bxe3 9. Qe2 Ne7 10. Qxe3 Nc6 11. Nf3 O-O 12. Rd1 Nf5 13. Qc5 Be6
14. Bd3 Qf6 15. c3 Rfc8 16. Qb6 Ne5 17. Be2 Nxf3+ 18. Bxf3 Qe7 19. O-O Rc6
20. Qb4 Qxb4 21. cxb4 Rd8 22. Nc5 b5 23. axb5 axb5 24. Rfe1 Nh4 25. Be2 Rb8
26. Nxe6 fxe6 27. g3 Ng6 28. Bd3 Ne7 29. Kg2 d4 30. Re5 Rcb6 31. Be4 Rd8
32. Kf3 g6 33. Rc5 h6 34. Bd3 Nd5 35. Bxg6 Nxb4 36. Rc7 d3 37. Ke4 d2
38. f4 Rdd6 39. Rh7 Rbc6 40. Re7 Rd3 41. Ke5 Re3+ 42. Kd4 Re1 43. Re8+ Kg7
44. Bd3 Rxd1 45. Bxb5 Rd6+ 46. Kc5 Rc1+ 47. Kxd6 d1=Q+ 48. Ke7 Rc7+
49. Kxe6 Qd5#
"""


def cp(info: dict, pov: chess.Color) -> int:
    return info["score"].pov(pov).score(mate_score=10000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--side", choices=("white", "black"), default="white")
    parser.add_argument("--threshold", type=int, default=80)
    args = parser.parse_args()

    side = chess.WHITE if args.side == "white" else chess.BLACK
    game = chess.pgn.read_game(io.StringIO(PGN))
    if game is None:
        raise SystemExit("could not parse PGN")

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "Hash": 128})
    limit = chess.engine.Limit(depth=args.depth)

    board = game.board()
    mistakes = []
    print(f"analysing as {args.side} at depth {args.depth}\n")
    print(f"{'move':>6} {'played':>8} {'eval before':>12} {'after':>8} {'drop':>7}  best")
    for move in game.mainline_moves():
        if board.turn != side:
            board.push(move)
            continue

        before_info = engine.analyse(board, limit)
        before = cp(before_info, side)
        best = before_info.get("pv", [None])[0]
        best_san = board.san(best) if best else "?"
        played_san = board.san(move)
        number = board.fullmove_number

        board.push(move)
        after = cp(engine.analyse(board, limit), side)
        drop = before - after

        flag = ""
        if drop >= args.threshold:
            flag = "  <-- MISTAKE"
            mistakes.append((number, played_san, before, after, drop, best_san))
        print(
            f"{number:>6} {played_san:>8} {before:>12} {after:>8} {drop:>7}  {best_san}{flag}"
        )

    engine.quit()

    print(f"\n{len(mistakes)} mistakes of {args.threshold}+ cp\n")
    mistakes.sort(key=lambda m: -m[4])
    print("worst first:")
    for number, played, before, after, drop, best in mistakes[:10]:
        print(
            f"  move {number:>3}: played {played:<7} {before:>6} -> {after:>6} "
            f"(lost {drop:>5} cp)   better: {best}"
        )


if __name__ == "__main__":
    main()
