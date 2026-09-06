"""Minimal playable agent backed by the experimental Numba move generator.

This module is intentionally separate from the stable root agent. The harness loads an
agent directory as a top-level ``agent`` module, so the package root is derived from this
file rather than from the process working directory.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from time import perf_counter
from types import ModuleType

import chess
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

bb_core = import_module("experiments.numba_engine.bb_core")

_numba_engine: ModuleType | None
_search_engine: ModuleType | None
try:
    _numba_engine = import_module("experiments.numba_engine.bb_numba")
    _search_engine = import_module("experiments.numba_engine.search")
except Exception as error:  # A legal python-chess fallback is safer than an init crash.
    _numba_engine = None
    _search_engine = None
    print(f"Numba engine unavailable; using legal fallback: {error!r}")

_FILES = "abcdefgh"
_PROMOTIONS = " nbrq"

_MAX_THINK_MS = 250.0
_MIN_SEARCH_MS = 5.0
_SEARCH_CLOCK_FRACTION = 0.02
_CONSERVATIVE_NODES_PER_MS = 500
_ABSOLUTE_NODE_LIMIT = 250_000


def _square_name(square: int) -> str:
    return _FILES[square & 7] + str((square >> 3) + 1)


def move_to_uci(move: int) -> str:
    """Convert the foundation's packed move representation to UCI."""
    origin = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    uci = _square_name(origin) + _square_name(target)
    return uci + (_PROMOTIONS[promotion] if promotion else "")


def _first_numba_move(fen: str) -> str:
    if _numba_engine is None:
        raise RuntimeError("Numba move generator did not initialize")
    board = np.array(bb_core.board_from_fen(fen), dtype=np.uint64)
    moves = np.empty(256, dtype=np.int32)
    move_count = int(_numba_engine.gen_legal(board, moves))
    if move_count == 0:
        raise ValueError("position has no legal moves")
    return move_to_uci(int(moves[0]))


def _search_limits(time_left_ms: int) -> tuple[int, int, float]:
    """Return maximum depth, total nodes, and soft search seconds."""
    remaining_ms = max(0, time_left_ms)
    reserve_ms = max(10, min(100, remaining_ms // 10))
    usable_ms = max(0, remaining_ms - reserve_ms)
    think_ms = min(_MAX_THINK_MS, usable_ms * _SEARCH_CLOCK_FRACTION)
    if think_ms < _MIN_SEARCH_MS:
        return 0, 0, 0.0
    node_limit = min(
        _ABSOLUTE_NODE_LIMIT,
        max(1, int(think_ms * _CONSERVATIVE_NODES_PER_MS)),
    )
    if _search_engine is None:
        return 0, 0, 0.0
    return int(_search_engine.MAX_SEARCH_DEPTH), node_limit, think_ms / 1_000.0


def _searched_numba_move(fen: str, time_left_ms: int) -> str:
    if _numba_engine is None or _search_engine is None:
        raise RuntimeError("Numba search did not initialize")

    board = _numba_engine.board_np(fen)
    legal_moves = np.empty(256, dtype=np.int32)
    move_count = int(_numba_engine.gen_legal(board, legal_moves))
    if move_count == 0:
        raise ValueError("position has no legal moves")

    # Always retain a legal move, and only replace it after a complete iteration.
    best_move = int(legal_moves[0])
    depth_limit, total_node_limit, soft_seconds = _search_limits(time_left_ms)
    if depth_limit == 0:
        return move_to_uci(best_move)

    started = perf_counter()
    nodes_used = 0
    transposition_table = _search_engine.new_transposition_table()
    for depth in range(1, depth_limit + 1):
        if perf_counter() - started >= soft_seconds:
            break
        remaining_nodes = total_node_limit - nodes_used
        if remaining_nodes <= 0:
            break
        state = _search_engine.new_state()
        move, _score, completed = _search_engine.search_root(
            board,
            depth,
            remaining_nodes,
            state,
            *transposition_table,
        )
        nodes_used += int(state[_search_engine.STATE_NODES])
        if not completed:
            break
        best_move = int(move)
    return move_to_uci(best_move)


def _legal_fallback(fen: str) -> str:
    board = chess.Board(fen)
    try:
        return next(iter(board.legal_moves)).uci()
    except StopIteration:
        raise ValueError("position has no legal moves") from None


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move for the official agent interface."""
    try:
        return _searched_numba_move(fen, time_left_ms)
    except Exception as error:
        print(f"Numba search failed; using move-generator fallback: {error!r}")
        try:
            return _first_numba_move(fen)
        except Exception as fallback_error:
            print(f"Numba move generation failed; using legal fallback: {fallback_error!r}")
            return _legal_fallback(fen)
