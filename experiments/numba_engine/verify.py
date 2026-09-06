"""Reproducible correctness checks for the experimental move generator.

Run from the repository root with::

    uv run python -m experiments.numba_engine.verify
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import bb_core, bb_movegen, bb_numba

if TYPE_CHECKING:
    import chess


@dataclass(frozen=True)
class PerftCase:
    """One standard perft position and its expected leaf count."""

    name: str
    fen: str
    depth: int
    expected: int


# The seven standard positions used by Stockfish's perft test script, at
# deliberately modest depths so the pure-Python reference remains practical.
STANDARD_CASES = (
    PerftCase(
        "start",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        4,
        197_281,
    ),
    PerftCase(
        "position_2",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        3,
        97_862,
    ),
    PerftCase(
        "position_3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        4,
        43_238,
    ),
    PerftCase(
        "position_4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        3,
        9_467,
    ),
    PerftCase(
        "position_5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        3,
        62_379,
    ),
    PerftCase(
        "position_6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        3,
        89_890,
    ),
    PerftCase(
        "position_7",
        "r7/4p3/5p1q/3P4/4pQ2/4pP2/6pp/R3K1kr w Q - 1 3",
        3,
        18_511,
    ),
)


def _numba_perft(fen: str, depth: int) -> int:
    board = np.array(bb_core.board_from_fen(fen), dtype=np.uint64)
    return int(bb_numba.perft(board, depth))


def run_standard_cases() -> bool:
    """Compare both implementations with fixed standard perft totals."""
    print("Standard perft cases")
    print("name             depth     expected       python        numba  result")
    passed = True
    for case in STANDARD_CASES:
        board = bb_core.board_from_fen(case.fen)
        python_total = bb_movegen.perft(board, case.depth)
        numba_total = _numba_perft(case.fen, case.depth)
        ok = python_total == case.expected and numba_total == case.expected
        passed &= ok
        result = "PASS" if ok else "FAIL"
        print(
            f"{case.name:<16} {case.depth:>5} {case.expected:>12} "
            f"{python_total:>12} {numba_total:>12}  {result}"
        )
    return passed


def _reference_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += _reference_perft(board, depth - 1)
        board.pop()
    return total


def _random_positions(chess_module: object, count: int, seed: int) -> list[str]:
    # Keep the import optional while retaining a small, deterministic corpus.
    chess_api = chess_module
    rng = random.Random(seed)
    positions: list[str] = []
    while len(positions) < count:
        board = chess_api.Board()
        for _ in range(80):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            if not board.is_game_over():
                positions.append(board.fen(en_passant="fen"))
                if len(positions) == count:
                    break
    return positions


def run_random_comparison(count: int, depth: int, seed: int) -> bool:
    """Compare both implementations with python-chess on reachable positions."""
    try:
        import chess
    except ImportError:
        print("Random comparison: SKIP (python-chess is unavailable)")
        return True

    print(f"Random comparison: {count} positions, depth {depth}, seed {seed}")
    for index, fen in enumerate(_random_positions(chess, count, seed), start=1):
        expected = _reference_perft(chess.Board(fen), depth)
        python_total = bb_movegen.perft(bb_core.board_from_fen(fen), depth)
        numba_total = _numba_perft(fen, depth)
        if python_total != expected or numba_total != expected:
            print(f"FAIL position {index}")
            print(f"FEN: {fen}")
            print(f"expected={expected} python={python_total} numba={numba_total}")
            return False
    print(f"Random comparison: PASS ({count}/{count})")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-positions", type=int, default=100)
    parser.add_argument("--random-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260_906)
    parser.add_argument("--skip-random", action="store_true")
    args = parser.parse_args(argv)
    if args.random_positions < 0:
        parser.error("--random-positions must be non-negative")
    if args.random_depth < 0:
        parser.error("--random-depth must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    standard_ok = run_standard_cases()
    random_ok = args.skip_random or run_random_comparison(
        args.random_positions,
        args.random_depth,
        args.seed,
    )
    if standard_ok and random_ok:
        print("Verification: PASS")
        return 0
    print("Verification: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
