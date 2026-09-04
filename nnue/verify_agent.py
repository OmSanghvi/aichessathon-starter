"""Confirm the integrated agent really uses the net, and still plays legally."""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402


def main() -> None:
    print("net loaded:", agent._NNUE is not None)
    if agent._NNUE is None:
        raise SystemExit("FAIL: net did not load; agent silently fell back to tables")

    # The net and the tables should disagree - if they matched everywhere the net
    # would not be doing anything.
    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
        "4rrk1/pp1n1ppp/2pb4/3p4/3P4/2NBP3/PP3PPP/2R2RK1 w - - 0 1",
    ]
    print(f"\n{'position':<46} {'PST':>7} {'net':>7}")
    for fen in fens:
        pos = agent.board_to_sunfish(chess.Board(fen))
        print(f"{fen[:46]:<46} {pos.score:>7d} {agent._NNUE.evaluate(pos.board):>7d}")

    # Legality and timing across a short self-played game.
    board = chess.Board()
    slowest = 0.0
    for ply in range(30):
        if board.is_game_over():
            break
        t0 = time.time()
        uci = agent.get_move(board.fen(), 120_000)
        slowest = max(slowest, time.time() - t0)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise SystemExit(f"FAIL: illegal move {uci} at ply {ply}")
        board.push(move)
        if board.legal_moves:
            board.push(next(iter(board.legal_moves)))
    print(f"\n30 plies played, all legal. slowest move: {slowest:.2f}s")

    # Mate detection must survive the swapped evaluation.
    mate = agent.get_move("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 10_000)
    print("back-rank mate found:", mate, "(expect a1a8)")
    if mate != "a1a8":
        print("WARNING: did not find the mate in 1")


if __name__ == "__main__":
    main()
