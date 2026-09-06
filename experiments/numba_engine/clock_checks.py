"""Clock-safety checks for the experimental Numba agent."""

from __future__ import annotations

import sys
from time import perf_counter

import chess

from . import agent

CLOCKS_MS = (1, 5, 25, 100, 500, 5_000, 120_000)


def main() -> int:
    # Import-time warm-up has completed before this point. Warm the Python wrapper path too.
    warmup_move = agent.get_move(chess.STARTING_FEN, 1_000)
    if chess.Move.from_uci(warmup_move) not in chess.Board().legal_moves:
        print(f"wrapper warm-up returned illegal move: {warmup_move}")
        return 1

    passed = True
    for clock_ms in CLOCKS_MS:
        started = perf_counter()
        move = agent.get_move(chess.STARTING_FEN, clock_ms)
        elapsed_ms = (perf_counter() - started) * 1_000.0
        legal = chess.Move.from_uci(move) in chess.Board().legal_moves
        within_clock = elapsed_ms < clock_ms
        ok = legal and within_clock
        passed &= ok
        depth, nodes, soft_seconds = agent._search_limits(clock_ms)
        print(
            f"clock={clock_ms:>6} ms elapsed={elapsed_ms:>8.3f} ms "
            f"depth={depth} nodes={nodes} soft={soft_seconds * 1_000:>6.1f} ms "
            f"move={move} {'PASS' if ok else 'FAIL'}"
        )

    if passed:
        print("Clock checks: PASS")
        return 0
    print("Clock checks: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
