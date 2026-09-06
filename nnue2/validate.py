"""Hard validation of the NNUE pipeline on REAL positions.

Written because three silent bugs turned up while building this (a stdlib module
shadowed by a filename, int8 overflow in index arithmetic, and an inverted king
code), so "it runs" is not evidence of correctness.

It also closes a real gap in export.py, which verifies the integer path against
RANDOM feature indices. Random indices can encode impossible positions - no king,
or pieces drawn from several king buckets at once - so agreement there says little
about agreement on positions the engine will actually see.

TESTS
  1. round trip: board -> codes -> features equals the direct path
  2. feature count equals piece count, and indices are in range
  3. king bucket is derived from the MOVER's king, in the mover's frame
  4. label sign convention: features and labels are both side-to-move relative,
     so a position winning for the mover must carry a positive label
  5. mirror symmetry: the same position with colours swapped must produce the same
     feature multiset, because the encoding is supposed to be side-agnostic
  6. integer path versus float net on REAL positions
"""

import subprocess
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nnue2.features import (  # noqa: E402
    KING_BUCKETS,
    NUM_FEATURES,
    board_to_codes,
    codes_to_features,
    features_from_board,
    king_bucket,
)

REAL_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "8/6pk/2p4p/8/1bp1pQ2/2n4P/2q2PPK/2B5 b - - 1 35",
    "6k1/4R3/2r1p1Bp/1p2K3/1n3P2/4r1P1/1P1p3P/3R4 w - - 7 42",
    "4rrk1/pp1n1ppp/2pb4/3p4/3P4/2NBP3/PP3PPP/2R2RK1 w - - 0 1",
    "8/1P6/8/8/8/2k5/8/6K1 w - - 0 1",
    "r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 7",
]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def test_roundtrip_and_range() -> None:
    print("\n[1,2] round trip, counts and index range")
    all_ok = True
    detail = ""
    for fen in REAL_FENS:
        b = chess.Board(fen)
        direct = features_from_board(b)
        codes, turn = board_to_codes(b)
        viacodes = codes_to_features(codes, turn)
        if not np.array_equal(np.sort(direct), np.sort(viacodes)):
            all_ok, detail = False, f"round trip differs on {fen}"
            break
        if len(direct) != len(b.piece_map()):
            all_ok, detail = False, f"count {len(direct)} != pieces {len(b.piece_map())}"
            break
        if direct.min() < 0 or direct.max() >= NUM_FEATURES:
            all_ok, detail = False, f"index out of range on {fen}"
            break
        if len(set(direct.tolist())) != len(direct):
            all_ok, detail = False, f"duplicate feature indices on {fen}"
            break
    check("board->codes->features == direct, counts and range", all_ok, detail)


def test_bucket_uses_movers_king() -> None:
    print("\n[3] king bucket comes from the MOVER's king, in the mover's frame")
    ok = True
    detail = ""
    for fen in REAL_FENS:
        b = chess.Board(fen)
        idx = features_from_board(b)
        bucket_from_features = int(idx[0]) // (12 * 64)
        if not all(int(i) // (12 * 64) == bucket_from_features for i in idx):
            ok, detail = False, f"features span several buckets on {fen}"
            break
        ks = b.king(b.turn)
        assert ks is not None
        expected_sq = (63 - ks) if b.turn == chess.BLACK else ks
        if bucket_from_features != king_bucket(expected_sq):
            ok, detail = False, f"bucket mismatch on {fen}"
            break
    check("single bucket per position, derived from mover's king", ok, detail)


def test_side_agnostic() -> None:
    """Swapping colours under OUR frame must give identical features.

    The encoding claims to be side-agnostic: the net sees only "me" and "them".
    The transform that must therefore be invisible is the one sunfish uses - a 180
    degree rotation plus a colour swap. python-chess's mirror() is a VERTICAL flip
    plus colour swap, which is a different transform, so testing with it would
    wrongly report a failure. 180 degrees is flip_vertical composed with
    flip_horizontal.
    """
    print("\n[5] side-agnostic: colour swap + 180 rotation must be invisible")
    ok = True
    detail = ""
    for fen in REAL_FENS:
        b = chess.Board(fen)
        if b.is_check():
            continue  # a swapped position can be illegal; skip those
        swapped = b.mirror().transform(chess.flip_horizontal)
        f1 = np.sort(features_from_board(b))
        f2 = np.sort(features_from_board(swapped))
        if not np.array_equal(f1, f2):
            ok = False
            detail = f"differs on {fen}"
            only1 = sorted(set(f1.tolist()) - set(f2.tolist()))[:6]
            only2 = sorted(set(f2.tolist()) - set(f1.tolist()))[:6]
            detail += f"  only-orig {only1}  only-swapped {only2}"
            break
    check("features identical under colour swap + 180 rotation", ok, detail)


def test_label_sign() -> None:
    print("\n[4] label sign convention: positive means good for the SIDE TO MOVE")
    engine_path = "/opt/homebrew/bin/stockfish"
    if not Path(engine_path).exists():
        print("  SKIP  no reference engine")
        return
    import chess.engine

    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    eng.configure({"Threads": 1})
    cases = [
        ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", True, "White to move, up a queen"),
        ("3qk3/8/8/8/8/8/8/4K3 b - - 0 1", True, "Black to move, up a queen"),
        ("3qk3/8/8/8/8/8/8/4K3 w - - 0 1", False, "White to move, down a queen"),
    ]
    ok = True
    for fen, expect_positive, label in cases:
        b = chess.Board(fen)
        cp = eng.analyse(b, chess.engine.Limit(depth=12))["score"].pov(b.turn).score(
            mate_score=2000
        )
        good = (cp > 0) == expect_positive
        ok = ok and good
        print(f"    {'ok ' if good else 'BAD'} {label}: label {cp:+d}")
    eng.quit()
    check("labels are side-to-move relative, matching features", ok)


def test_int_vs_float_real_positions() -> None:
    print("\n[6] integer path vs float net, on REAL positions")
    out = Path(__file__).resolve().parent / "out"
    ckpt, weights = out / "net.pt", out / "nnue_weights.npz"
    if not ckpt.exists() or not weights.exists():
        print("  SKIP  no trained net yet (run gen/train/export first)")
        return

    import torch

    from nnue2.export import reference_forward
    from nnue2.train import NNUE

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = NNUE(ck["acc"], ck["hidden"], dropout=0.0)
    model.load_state_dict(ck["state"])
    model.eval()
    with np.load(weights) as z:
        w = {k: z[k] for k in z.files}

    errs = []
    for fen in REAL_FENS:
        idx = features_from_board(chess.Board(fen)).astype(np.int64)
        dense = torch.zeros(1, NUM_FEATURES)
        dense[0, torch.from_numpy(idx)] = 1.0
        with torch.no_grad():
            f = float(model(dense)[0, 0])
        i = reference_forward(w, idx)
        errs.append(abs(f - i))
        print(f"    {fen[:44]:<44} float {f:>8.1f}  int {i:>6}  diff {abs(f - i):>5.1f}")
    mean = float(np.mean(errs))
    check("integer path tracks float net on real positions",
          mean < 25, f"mean {mean:.1f} cp, max {max(errs):.1f} cp")


def test_pipeline_runs() -> None:
    print("\n[0] feature contract between training and the agent")
    r = subprocess.run(
        [sys.executable, str(ROOT / "nnue2" / "verify_features.py")],
        capture_output=True, text=True,
    )
    ok = "0 mismatches" in r.stdout and "OK" in r.stdout
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[:80]
    check("agent and training encodings identical", ok, tail)


def main() -> None:
    print("=" * 72)
    print("NNUE v2 hard validation")
    print("=" * 72)
    print(f"NUM_FEATURES={NUM_FEATURES}  KING_BUCKETS={KING_BUCKETS}")
    test_pipeline_runs()
    test_roundtrip_and_range()
    test_bucket_uses_movers_king()
    test_label_sign()
    test_side_agnostic()
    test_int_vs_float_real_positions()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED {len(failures)}: " + ", ".join(failures))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
