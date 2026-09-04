"""Guard: assert the submission zip contains only what we intend to ship.

A labelling engine now exists on this machine. The rules allow training on
engine-annotated data but forbid shipping an engine, so this check exists to make
the boundary mechanical rather than a thing we remember. It also catches the
subtler packaging mistake: harness/package.py globs every root-level *.py, so a
stray script at the repo root would silently ride along.

Run: .venv/bin/python nnue/check_submission.py [path/to/submission.zip]
"""

import sys
import zipfile
from pathlib import Path

# Exactly what may appear in the archive. Weights get added here when a net ships.
ALLOWED = {"agent.py"}
ALLOWED_PREFIXES = ("weights/",)

# Anything matching these is an immediate failure.
BANNED_SUBSTRINGS = ("stockfish", "lc0", "maia", "leela")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "submission.zip")
    if not path.exists():
        print(f"FAIL: {path} does not exist")
        return 1

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bad_crc = archive.testzip()

    problems = []
    if bad_crc is not None:
        problems.append(f"corrupt member: {bad_crc}")

    if "agent.py" not in names:
        problems.append("agent.py missing from the archive root")

    for name in names:
        lowered = name.lower()
        for banned in BANNED_SUBSTRINGS:
            if banned in lowered:
                problems.append(f"BANNED engine artefact in zip: {name}")
        if name in ALLOWED or name.startswith(ALLOWED_PREFIXES):
            continue
        problems.append(f"unexpected member: {name}")

    # Nested directories for agent.py break the platform's `import agent`.
    if any(n.endswith("/agent.py") for n in names):
        problems.append("agent.py is nested in a folder; it must sit at the zip root")

    print(f"{path}: {len(names)} member(s)")
    for name in sorted(names):
        print(f"  {name}")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK: archive contains only intended files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
