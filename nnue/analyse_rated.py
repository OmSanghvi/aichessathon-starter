"""Analyse the rated-game log from the platform.

The interesting columns are the clock ones. A prediction was made from simulation
that the live time formula ends a long game with about 1.4s on the clock; this is
where that gets checked against reality, and where we find out whether any game
came close to flagging - a flag is an automatic loss and the easiest one to
self-inflict.
"""

import io
import csv

RAW = """round,colour,opponent,result,termination,init_s,moves,time_used_s,slowest_s,fastest_s,average_s,clock_left_s
Rated 14,White,OpenAnand,Win,Checkmate,0.8,96,166.8,12.3,0.3,1.7,1.2
Rated 13,White,Dio,Draw,Threefold repetition,0.8,39,115.6,11.3,0.0,3.0,23.9
Rated 12,Black,lefischer,Loss,Checkmate,0.8,70,152.9,15.6,0.0,2.2,2.1
Rated 11,White,Fight Club,Loss,Checkmate,0.8,42,128.1,11.7,0.9,3.0,12.9
Rated 10,Black,YouU-AreR,Win,Checkmate,0.8,59,145.7,9.9,0.7,2.5,3.8
Rated 9,Black,A weapon to surpass Metal Gear,Win,Checkmate,,61,145.9,8.7,0.7,2.4,4.6
Rated 8,Black,SOLO,Win,Checkmate,,39,96.8,3.6,1.6,2.5,42.7
Rated 7,White,Baryon,Draw,Threefold repetition,,54,119.1,3.7,1.2,2.2,27.9
Rated 6,Black,claudeshark,Loss,Checkmate,,52,115.8,3.6,1.3,2.2,30.2
Rated 5,White,Trio Duo,Win,Checkmate,,28,76.3,3.6,2.1,2.7,57.7
Rated 4,White,Columbia Trader,Win,Checkmate,,25,70.4,3.5,2.2,2.8,62.1
Rated 3,Black,Tempo,Win,Checkmate,,55,120.6,3.7,1.3,2.2,26.9
Rated 2,White,e=mc^2,Loss,Checkmate,,24,67.9,3.6,2.2,2.8,64.1
Rated 1,Black,Mate in One,Loss,Checkmate,,15,46.8,3.6,2.6,3.1,80.7
"""


def main() -> None:
    rows = list(csv.DictReader(io.StringIO(RAW)))
    for r in rows:
        r["n"] = int(r["round"].split()[1])
        r["moves"] = int(r["moves"])
        for k in ("time_used_s", "slowest_s", "fastest_s", "average_s", "clock_left_s"):
            r[k] = float(r[k])
    rows.sort(key=lambda r: r["n"])

    w = sum(r["result"] == "Win" for r in rows)
    d = sum(r["result"] == "Draw" for r in rows)
    losses = sum(r["result"] == "Loss" for r in rows)
    print(f"OVERALL: {w}W {losses}L {d}D over {len(rows)} games "
          f"= {(w + d / 2) / len(rows):.1%}\n")

    # The slowest-move column splits the log cleanly into two deployed versions.
    old = [r for r in rows if r["slowest_s"] < 5]
    new = [r for r in rows if r["slowest_s"] >= 5]
    for label, group in (("rounds with slowest < 5s", old), ("rounds with slowest >= 5s", new)):
        gw = sum(r["result"] == "Win" for r in group)
        gd = sum(r["result"] == "Draw" for r in group)
        gl = sum(r["result"] == "Loss" for r in group)
        rounds = ",".join(str(r["n"]) for r in group)
        print(f"{label}: rounds {rounds}")
        print(f"   {gw}W {gl}L {gd}D = {(gw + gd / 2) / len(group):.1%}   "
              f"slowest {min(r['slowest_s'] for r in group):.1f}-"
              f"{max(r['slowest_s'] for r in group):.1f}s   "
              f"min clock left {min(r['clock_left_s'] for r in group):.1f}s")
    print()

    print("FLAG RISK: clock left, sorted (a flag is an automatic loss)")
    for r in sorted(rows, key=lambda r: r["clock_left_s"])[:7]:
        budget = 120 + r["moves"] * 0.5
        print(f"   round {r['n']:>2} {r['result']:<5} {r['moves']:>3} moves  "
              f"used {r['time_used_s']:>6.1f}s of {budget:>5.1f}s  "
              f"left {r['clock_left_s']:>5.1f}s  fastest {r['fastest_s']:.1f}s")
    print()

    print("LONGER GAME -> LESS CLOCK LEFT?")
    for r in sorted(rows, key=lambda r: -r["moves"])[:6]:
        print(f"   {r['moves']:>3} moves -> {r['clock_left_s']:>5.1f}s left "
              f"({r['result']}, avg {r['average_s']:.1f}s/move)")


if __name__ == "__main__":
    main()
