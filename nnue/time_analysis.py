"""Where does the clock actually go, and does the budget front-load?

Observed symptom: the bot plays slowly early and blunders late. The lost game's
own clock readings let us measure the spend rate directly, and then simulate
candidate budget formulas over a long game to see which ones still have time left
when the endgame arrives.

Time control is 120s + 0.5s per move, per side.
"""

BASE_MS = 120_000
INCREMENT_MS = 500
DELAY_MS = 200

# White's clock from the lost game, at the listed move number.
OBSERVED = [(8, 113.005), (20, 57.625), (30, 32.985), (40, 22.504), (49, 12.904)]


def observed_spend() -> None:
    print("actual spend rate in the lost game (White = our bot)\n")
    print(f"{'moves':>12}{'clock drop':>12}{'earned':>9}{'spent':>9}{'per move':>10}")
    for (m0, c0), (m1, c1) in zip(OBSERVED, OBSERVED[1:], strict=False):
        moves = m1 - m0
        earned = moves * INCREMENT_MS / 1000
        spent = (c0 - c1) + earned
        print(
            f"{m0:>4}->{m1:<6}{c0 - c1:>11.1f}s{earned:>8.1f}s"
            f"{spent:>8.1f}s{spent / moves:>9.2f}s"
        )
    print("\nfront-loading: ~5s a move early, ~1.5s a move once it matters.")


def budget(remaining_ms: float, divisor: float, multiplier: float) -> tuple[float, float]:
    """Return (soft_ms, think_ms) for a candidate formula."""
    remaining = max(remaining_ms - DELAY_MS, 0)
    base = remaining / divisor + INCREMENT_MS
    soft = max(min(base, remaining / 4), 20)
    think = max(min(multiplier * base, remaining / 2), 40)
    return soft, think


def simulate(divisor: float, multiplier: float, moves: int = 80, use_think: float = 0.7) -> dict:
    """Play out a game spending `use_think` of the way between soft and think.

    Real searches overshoot soft and finish an iteration somewhere before think,
    so a fraction between the two models the spend better than either endpoint.
    """
    clock = float(BASE_MS)
    spends = []
    flagged_at = None
    for move in range(1, moves + 1):
        if clock <= 0:
            flagged_at = move
            break
        soft, think = budget(clock, divisor, multiplier)
        spend = soft + (think - soft) * use_think
        spend = min(spend, max(clock - DELAY_MS, 0))
        clock -= spend
        clock += INCREMENT_MS
        spends.append(spend)
    return {
        "divisor": divisor,
        "multiplier": multiplier,
        "flagged_at": flagged_at,
        "spends": spends,
        "clock_left": clock,
    }


def report(results: list[dict]) -> None:
    print("\n\nsimulated spend per move (seconds) under candidate formulas")
    print("columns are move numbers; 'left' is clock after 80 moves\n")
    marks = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    header = f"{'divisor':>8}{'mult':>6}" + "".join(f"{m:>7}" for m in marks) + f"{'left':>9}"
    print(header)
    for r in results:
        row = f"{r['divisor']:>8.0f}{r['multiplier']:>6.1f}"
        for m in marks:
            if m <= len(r["spends"]):
                row += f"{r['spends'][m - 1] / 1000:>7.2f}"
            else:
                row += f"{'-':>7}"
        flag = "FLAG" if r["flagged_at"] else f"{r['clock_left'] / 1000:>8.1f}s"
        row += f"{flag:>9}"
        print(row)


def main() -> None:
    observed_spend()
    results = []
    for divisor, multiplier in (
        (40, 5.0),  # current
        (40, 2.0),
        (60, 3.0),
        (60, 2.0),
        (80, 2.0),
        (30, 1.5),
    ):
        results.append(simulate(divisor, multiplier))
    report(results)
    print("\nGoal: avoid burning the bank early so the late middlegame and endgame,")
    print("where the measured blunders happen, still get a usable budget.")


if __name__ == "__main__":
    main()
