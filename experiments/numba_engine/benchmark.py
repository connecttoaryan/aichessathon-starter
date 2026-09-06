"""Short, reproducible benchmark for the experimental Numba foundation.

Run from the repository root with::

    uv run python -m experiments.numba_engine.benchmark
"""

from __future__ import annotations

import argparse
import platform
import sys
from time import perf_counter

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EXPECTED_START_PERFT = {
    1: 20,
    2: 400,
    3: 8_902,
    4: 197_281,
    5: 4_865_609,
    6: 119_060_324,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="starting-position perft depth for the node-rate sample (default: 5)",
    )
    args = parser.parse_args(argv)
    if args.depth < 1:
        parser.error("--depth must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import_started = perf_counter()
    from . import bb_numba

    cold_import_seconds = perf_counter() - import_started

    import numba
    import numpy as np

    board = bb_numba.board_np(START_FEN)
    perft_started = perf_counter()
    nodes = int(bb_numba.perft(board, args.depth))
    perft_seconds = perf_counter() - perft_started
    nodes_per_second = nodes / perft_seconds

    expected = EXPECTED_START_PERFT.get(args.depth)
    correct = expected is None or nodes == expected

    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"numba: {numba.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"cold import + JIT: {cold_import_seconds:.3f} s")
    print(f"perft depth: {args.depth}")
    print(f"nodes: {nodes}")
    if expected is not None:
        print(f"expected: {expected}")
    print(f"perft time: {perft_seconds:.3f} s")
    print(f"node rate: {nodes_per_second:,.0f} nodes/s")
    print(f"result: {'PASS' if correct else 'FAIL'}")
    return 0 if correct else 1


if __name__ == "__main__":
    sys.exit(main())
