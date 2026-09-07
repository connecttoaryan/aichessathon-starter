# EXPERIMENTAL Numba chess bot — NOT SUBMISSION-READY

This package contains a correctness-first bitboard move generator and a separately playable
experimental bot. It is not integrated into the stable root `agent.py`, has not passed platform
validation, and must not be treated as the repository's submission agent.

## Contents

- `bb_core.py` builds attack tables, represents board state, parses FEN, and provides the
  pure-Python bitboard primitives.
- `bb_movegen.py` is the readable pure-Python move generator, move application code, legality
  filter, and perft reference.
- `bb_numba.py` implements the same hot path with NumPy arrays and Numba `njit` compilation.
- `search.py` provides bounded material negamax, alpha-beta pruning, iterative-deepening support,
  capture quiescence, move ordering, and a fixed-size exact-checked transposition table.
- `agent.py` exposes `get_move(fen: str, time_left_ms: int) -> str` for local harness games.
- `verify.py` checks both implementations against fixed perft totals and, when python-chess is
  available, a bounded deterministic set of reachable positions.
- `agent_checks.py` checks special moves, mate finding, node aborts, quiescence, and TT bounds.
- `clock_checks.py` checks legal replies and elapsed time over clocks from 1 ms to 120 seconds.
- `benchmark.py` measures fresh-process import/JIT time and a short legal-perft node rate.

`experiments` and `experiments.numba_engine` are regular Python packages. Run their tools from
the repository root with `python -m`; they do not depend on the current directory accidentally
exposing sibling modules as top-level imports.

## Verification

Install the repository environment once with `uv sync`, then run:

```console
uv run python -m experiments.numba_engine.verify
uv run python -m experiments.numba_engine.agent_checks
uv run python -m experiments.numba_engine.clock_checks
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

The latest local results were:

- Seven of seven fixed standard perft cases passed in both implementations.
- 100 of 100 deterministic reachable positions matched python-chess at depth 2.
- Ordinary, promotion, castling, en-passant, two mate-in-one, one-node abort, bounded-TT, and
  quiescence-horizon checks all passed.
- Clock checks returned legal moves within the supplied clock at 1, 5, 25, 100, 500, 5,000, and
  120,000 ms. The measured 1 ms reply took 0.053 ms locally.

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
| Complete experimental-agent cold import plus JIT | 23.248 s |

The count matched the expected total. These timings are local measurements, not promises about
the competition CPU. The reported node rate is legal perft throughput; it is not search NPS and
does not include evaluation or search overhead. Import remains below the current 90-second
competition initialization budget locally, but the platform validation log is authoritative.

## Playable-agent design

`agent.py` first asks the compiled move generator for a legal move, so it always has a candidate
before search begins. Iterative deepening replaces that candidate only after a fully completed
iteration. If compiled search fails, it falls back to the Numba legal generator; if that also
fails, it asks python-chess for a legal move.

The search has hard limits of depth 4, ply 64, and eight quiescence plies. A per-move node budget
is derived conservatively from 2% of the clock after a reserve, capped at 250 ms and 250,000
nodes. Clocks too small to provide a 5 ms search allowance skip search. A soft wall-time check
runs between iterations, while the node counter bounds every compiled recursive call.

Evaluation is material only: pawn 100, knight 320, bishop 330, rook 500, queen 900. It has no
piece-square tables, mobility, pawn structure, king safety, tempo, or draw-aware scoring.
Quiescence searches captures and all legal evasions while in check.

Move ordering is an exact-verified transposition-table move, then captures, then quiet moves. The
direct-mapped TT has 2,048 slots, stores the complete 16-word board to reject hash collisions,
occupies 284,672 bytes, and is newly allocated for each move. Mate scores are not cached because
they have not been normalized for ply.

## Bounded arena results

Command shape:

```console
uv run python -m harness.arena --agent experiments/numba_engine --opponent baselines/<name> --games 2 --base-ms 10000 --increment-ms 100 --ply-cap 200
```

Each matchup used one game as White and one as Black. This sample is a stability smoke test, not
a statistically meaningful strength estimate.

| Opponent | Experimental White | Experimental Black | Score | Terminations | Failed terminations |
|---|---|---|---:|---|---|
| `baselines/random` | win | win | +2 =0 -0 | checkmate 2 | crash 0, flag 0, illegal 0, init 0 |
| `baselines/greedy` | win | win | +2 =0 -0 | checkmate 2 | crash 0, flag 0, illegal 0, init 0 |
| `baselines/minimax` | win | win | +2 =0 -0 | checkmate 2 | crash 0, flag 0, illegal 0, init 0 |
| `agent_claude.py` at `d764e3f` | 0-1-4 | 0-0-5 | +0 =1 -9, 5.0% | checkmate 9, threefold 1 | crash 0, flag 0, illegal 0, init 0 |

The ten-game `agent_claude.py` match used the harness default 10 s + 0.1 s clock. The Numba bot
lost games 1, 2, and 4–10 by checkmate and drew game 3 by threefold repetition. It scored 0-1-4 as
White and 0-0-5 as Black. This is the fixed pre-evaluation benchmark for commit `e80c308`.

For a reproducible teammate comparison, put the exact reference file in a temporary directory as
`agent.py`, then run:

```console
uv run python -m harness.arena --agent experiments/numba_engine --opponent <temporary-folder-path> --games 10
```

The reference used above is
[`agent_claude.py` at commit `d764e3f`](https://github.com/connecttoaryan/aichessathon-starter/blob/d764e3f3ed26fc12cd0082e49dcf59e89abef282/agent_claude.py).
Future strength comparisons must also preserve the prior Numba bot from commit `e80c308` in a
temporary folder and use the same harness command with `--games 20`. Do not compare two mutable
working directories and call the result reproducible.

A separate two-game low-clock smoke test against random at 100 ms + 10 ms also finished with two
checkmate wins and zero crashes, flags, illegal moves, or init failures.

## Aborted tapered-evaluation experiment (2026-09-07)

A Numba-compatible tapered evaluator was implemented locally with distinct midgame/endgame
values and piece-square tables, phase interpolation, bishop-pair, passed/isolated/doubled-pawn,
rook-file, mobility, and king-safety terms. Its targeted symmetry and obvious-position checks
passed. The full seven-position perft suite, 100-position python-chess comparison, special-move
and mate checks, node/TT/quiescence guards, and clock checks also passed. Two-game random and
greedy smoke matches each finished 2-0 by checkmate.

The change was not retained because fresh-process initialization was unreliable:

- The first two-game minimax run lost both games by `init`. Removing a redundant evaluator
  warm-up produced one checkmate win and one threefold draw on the single permitted rerun.
- One cold agent import measured 60.631 s after that change, but another form of the evaluator
  measured 224.817 s wall / 109.797 s CPU, beyond the 90-second contract even when judging by
  CPU time.
- The required 20-game comparison against prior Numba commit `e80c308` produced one genuine
  threefold draw, followed by 19 `both_failed` initialization terminations. The harness summary
  (`+0 =20 -0`, 50.0%) is therefore an invalid strength result: 19 nominal draws were void games.
- A module breakdown under the same local conditions measured 66.109 CPU seconds for the
  unchanged `bb_numba` warm-up, 6.422 for the evaluator, and 14.969 for search import. Removing
  the verification-only perft warm-up made compilation worse (129.172 CPU seconds total), so
  that experiment was also reverted.

Per the reliability rule, the tapered evaluator was reverted completely and search-quality work
was not started. No post-change match against `agent_claude.py` was run after this blocker. The
engine documented in the rest of this README therefore remains the material-evaluation build at
`e80c308`, not a stronger positional version.

## Competition-environment behavior

All `njit` functions use `cache=False`. Each fresh process compiles in memory during the import
warm-up and does not need to create Numba cache files beside the read-only source. This also means
there is no cross-game compiled-cache reuse, which is appropriate because every competition game
starts in a fresh container with an empty `/tmp`.

The implementation uses only NumPy and Numba at runtime. The optional random verification also
uses python-chess. These packages are all part of the fixed competition environment. No network,
subprocess, GPU, or multi-core behavior is used.

## Known limitations

- There is no repetition history or draw-aware search scoring.
- The material-only evaluation is strategically weak and frequently treats unrelated moves as
  equal.
- The Python wall clock cannot interrupt a currently executing Numba call. Safety depends on the
  strict node, depth, ply, and quiescence limits described above.
- FEN input is assumed to be valid and internally consistent; it is not validated before use.
- Move generation uses fixed 256-entry buffers and copies the board array at each node. This is a
  correctness-focused experiment, not a final optimized engine design.
- Only orthodox chess castling is represented; Chess960 is outside this experiment's scope.
- Local perft coverage is strong but cannot prove correctness for every legal or malformed
  position.
- Only small local arena samples have been run, all from the standard starting position.
- The complete experimental agent has not been tested in the competition's Linux container or
  accepted by platform validation.

The experimental interface is playable, but this package remains **NOT SUBMISSION-READY**. Do not
replace the stable root `agent.py` with it based on these bounded local results.
