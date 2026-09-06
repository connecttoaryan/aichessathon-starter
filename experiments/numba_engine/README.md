# Experimental Numba bitboard foundation

This package is a correctness-first bitboard move generator for experiments. It is not a
submission-ready chess agent and is not integrated into `agent.py`.

## Contents

- `bb_core.py` builds attack tables, represents board state, parses FEN, and provides the
  pure-Python bitboard primitives.
- `bb_movegen.py` is the readable pure-Python move generator, move application code, legality
  filter, and perft reference.
- `bb_numba.py` implements the same hot path with NumPy arrays and Numba `njit` compilation.
- `verify.py` checks both implementations against fixed perft totals and, when python-chess is
  available, a bounded deterministic set of reachable positions.
- `benchmark.py` measures fresh-process import/JIT time and a short legal-perft node rate.

`experiments` and `experiments.numba_engine` are regular Python packages. Run their tools from
the repository root with `python -m`; they do not depend on the current directory accidentally
exposing sibling modules as top-level imports.

## Verification

Install the repository environment once with `uv sync`, then run:

```console
uv run python -m experiments.numba_engine.verify
```

The default run checks seven standard perft positions at bounded depths in both the Python and
Numba implementations. The positions come from the
[official Stockfish perft script](https://github.com/official-stockfish/Stockfish/blob/master/tests/perft.sh).
It then checks 100 deterministic reachable positions at depth 2 against python-chess, using seed
`20260906`. Every case prints expected and actual totals, and any mismatch exits nonzero.

Useful bounded variants:

```console
uv run python -m experiments.numba_engine.verify --skip-random
uv run python -m experiments.numba_engine.verify --random-positions 25 --random-depth 2
uv run ruff check experiments/numba_engine experiments/__init__.py
```

The Numba module deliberately warms the complete compiled call graph during import. A fresh
verification process can therefore be quiet for several seconds before printing its table.

## Benchmark

Run one fresh-process benchmark from the repository root:

```console
uv run python -m experiments.numba_engine.benchmark --depth 5
```

Local measurement on 2026-09-06:

| Measurement | Result |
|---|---:|
| Environment | Windows 11, Python 3.12.4, Numba 0.67.0, NumPy 2.5.2 |
| Cold process import plus JIT warm-up | 22.719 s |
| Starting-position perft depth 5 | 4,865,609 nodes |
| Timed perft | 1.220 s |
| Perft node rate | 3,986,869 nodes/s |

The count matched the expected total. These timings are local measurements, not promises about
the competition CPU. The reported node rate is legal perft throughput; it is not search NPS and
does not include evaluation or search overhead. Import remains below the current 90-second
competition initialization budget locally, but the platform validation log is authoritative.

## Competition-environment behavior

All `njit` functions use `cache=False`. Each fresh process compiles in memory during the import
warm-up and does not need to create Numba cache files beside the read-only source. This also means
there is no cross-game compiled-cache reuse, which is appropriate because every competition game
starts in a fresh container with an empty `/tmp`.

The implementation uses only NumPy and Numba at runtime. The optional random verification also
uses python-chess. These packages are all part of the fixed competition environment. No network,
subprocess, GPU, or multi-core behavior is used.

## Known limitations

- There is no `get_move(fen, time_left_ms)` entry point.
- There is no encoded-move-to-UCI conversion.
- There is no search, evaluation, move ordering, quiescence, transposition table, or time
  management.
- There is no repetition history or complete game-result/draw handling.
- FEN input is assumed to be valid and internally consistent; it is not validated before use.
- Move generation uses fixed 256-entry buffers and copies the board array at each node. This is a
  correctness foundation, not a final optimized engine design.
- Only orthodox chess castling is represented; Chess960 is outside this experiment's scope.
- Local perft coverage is strong but cannot prove correctness for every legal or malformed
  position.

Do not submit this package on its own. Complete search, evaluation, time control, UCI conversion,
and `get_move` integration are still required before it can become an agent.
