"""Compare time-budget formulas by simulation before spending game time on them.

The current formula burns 118 of 120 seconds in the first 20 moves and then plays
the rest of the game on the increment, which is where the measured blunders are.

Candidates follow IDEAS.md - budget from the clock you were handed divided by the
moves you expect to have left - and deliberately avoid hardcoding the 0.5s
increment. Dropping that constant makes the formula scale-invariant, which matters
for a practical reason as much as a principled one: a formula with an absolute
constant in it cannot be tested at a fast time control, because the constant stops
being proportional. Without it, a 12s game is a faithful scale model of a 120s one.

Success looks like: usable time in the move 25-60 range, no collapse onto the
increment, and a safety buffer that never approaches a flag.
"""

BASE_MS = 120_000
INCREMENT_MS = 500
DELAY_MS = 200
OVERSHOOT = 0.7  # searches finish an iteration somewhere between soft and hard


def current(remaining_ms: float, move_no: int) -> tuple[float, float]:
    rem = max(remaining_ms - DELAY_MS, 0)
    base = rem / 40 + 500
    return max(min(base, rem / 4), 20), max(min(5 * base, rem / 2), 40)


def moves_left_estimate(move_no: int) -> int:
    """Assume a long-ish game but never fewer than 20 moves to come."""
    return max(20, 60 - move_no)


def candidate(remaining_ms: float, move_no: int, mult: float, floor: int) -> tuple[float, float]:
    rem = max(remaining_ms - DELAY_MS, 0)
    left = max(floor, 60 - move_no)
    base = rem / left
    soft = max(min(base, rem / 4), 20)
    hard = max(min(mult * base, rem / 3), 40)
    return soft, hard


def simulate(fn, label: str, moves: int = 90) -> dict:
    clock = float(BASE_MS)
    spends, clocks = [], []
    for mv in range(1, moves + 1):
        if clock <= DELAY_MS:
            return {"label": label, "flagged": mv, "spends": spends, "clocks": clocks}
        soft, hard = fn(clock, mv)
        spend = min(soft + (hard - soft) * OVERSHOOT, max(clock - DELAY_MS, 0))
        clocks.append(clock)
        spends.append(spend)
        clock += INCREMENT_MS - spend
    return {"label": label, "flagged": None, "spends": spends, "clocks": clocks, "left": clock}


def report(results: list[dict]) -> None:
    marks = [1, 5, 10, 20, 30, 40, 50, 60, 75, 90]
    print("\nSPEND per move, seconds")
    print(f"{'formula':<22}" + "".join(f"{m:>7}" for m in marks) + f"{'left':>9}")
    for r in results:
        row = f"{r['label']:<22}"
        for m in marks:
            row += f"{r['spends'][m - 1] / 1000:>7.2f}" if m <= len(r["spends"]) else f"{'-':>7}"
        row += "  FLAG" if r["flagged"] else f"{r['left'] / 1000:>8.1f}s"
        print(row)

    print("\nCLOCK REMAINING at that move, seconds  (how close to a flag)")
    print(f"{'formula':<22}" + "".join(f"{m:>7}" for m in marks))
    for r in results:
        row = f"{r['label']:<22}"
        for m in marks:
            row += f"{r['clocks'][m - 1] / 1000:>7.1f}" if m <= len(r["clocks"]) else f"{'-':>7}"
        print(row)


def main() -> None:
    results = [simulate(current, "current (/40, x5)")]
    for mult, floor in ((2.0, 20), (1.5, 20), (2.0, 26), (1.5, 30)):
        results.append(
            simulate(
                lambda c, m, mult=mult, floor=floor: candidate(c, m, mult, floor),
                f"/max({floor},60-n) x{mult}",
            )
        )
    report(results)
    print("\nMinimum clock seen (a small number here is a flag risk):")
    for r in results:
        print(f"  {r['label']:<22} {min(r['clocks']) / 1000:>7.1f}s")


if __name__ == "__main__":
    main()
