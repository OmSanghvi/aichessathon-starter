SHELL := /bin/bash

.PHONY: setup play arena zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -c 'import zipfile; archive = zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED); archive.write("agent.py"); archive.close()'

gate:
	uv run ruff check agent.py nengine
	uv run mypy agent.py
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
