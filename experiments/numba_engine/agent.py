"""Minimal playable agent backed by the experimental Numba move generator.

This module is intentionally separate from the stable root agent. The harness loads an
agent directory as a top-level ``agent`` module, so the package root is derived from this
file rather than from the process working directory.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
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


def _searched_numba_move(fen: str) -> str:
    if _numba_engine is None or _search_engine is None:
        raise RuntimeError("Numba search did not initialize")

    board = _numba_engine.board_np(fen)
    legal_moves = np.empty(256, dtype=np.int32)
    move_count = int(_numba_engine.gen_legal(board, legal_moves))
    if move_count == 0:
        raise ValueError("position has no legal moves")

    # Always retain a legal move, and only replace it after a complete iteration.
    best_move = int(legal_moves[0])
    for depth in range(1, int(_search_engine.MAX_SEARCH_DEPTH) + 1):
        state = _search_engine.new_state()
        move, _score, completed = _search_engine.search_root(
            board,
            depth,
            int(_search_engine.DEFAULT_NODE_LIMIT),
            state,
        )
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
    del time_left_ms  # Stage 3 will derive the node budget from the remaining clock.
    try:
        return _searched_numba_move(fen)
    except Exception as error:
        print(f"Numba search failed; using move-generator fallback: {error!r}")
        try:
            return _first_numba_move(fen)
        except Exception as fallback_error:
            print(f"Numba move generation failed; using legal fallback: {fallback_error!r}")
            return _legal_fallback(fen)
