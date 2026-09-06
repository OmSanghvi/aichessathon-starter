"""Assemble the single-file agent.py from the verified engine modules.

The submission must be ONE file, but developing and testing a move generator inside
a monolith is miserable, so the engine lives in nengine/ with its own perft and
search tests and this script concatenates it. Generating agent.py rather than
hand-copying means the shipped file can never drift from the tested modules.

  .venv/bin/python nengine/build_agent.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "nengine" / "agent_header.py"
FOOTER = ROOT / "nengine" / "agent_footer.py"
BOARD = ROOT / "nengine" / "board.py"
SEARCH = ROOT / "nengine" / "search.py"
OUT = ROOT / "agent.py"


def body_from(path: Path, start_marker: str) -> str:
    """Everything from `start_marker` onward, i.e. skip the module's imports."""
    src = path.read_text()
    idx = src.index(start_marker)
    return src[idx:].rstrip() + "\n"


def main() -> None:
    parts = [
        HEADER.read_text().rstrip() + "\n",
        "\n# " + "=" * 74 + "\n"
        "# Board representation and move generation (perft-verified against\n"
        "# python-chess; see nengine/test_perft.py)\n"
        "# " + "=" * 74 + "\n\n",
        body_from(BOARD, "def initial_board"),
        "\n# " + "=" * 74 + "\n"
        "# Search: alpha-beta, transposition table, quiescence, move ordering\n"
        "# " + "=" * 74 + "\n\n",
        body_from(SEARCH, "MATE = 30000"),
        "\n",
        FOOTER.read_text().rstrip() + "\n",
    ]
    OUT.write_text("".join(parts))
    lines = OUT.read_text().count("\n")
    print(f"wrote {OUT} ({lines} lines)")
    print("now run: uv run ruff check agent.py && uv run mypy agent.py")


if __name__ == "__main__":
    main()
