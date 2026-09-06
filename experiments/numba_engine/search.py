"""Bounded material-only negamax for the experimental Numba agent."""

from __future__ import annotations

import numpy as np
from numba import njit

from . import bb_numba

INFINITY = 1_000_000
MATE_SCORE = 100_000
MAX_SEARCH_DEPTH = 4
MAX_PLY = 64
DEFAULT_NODE_LIMIT = 250_000

STATE_NODES = 0
STATE_ABORTED = 1

PIECE_VALUES = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@njit(cache=False)
def popcount(bitboard):
    count = 0
    while bitboard:
        bitboard &= bitboard - bb_numba.ONE
        count += 1
    return count


@njit(cache=False)
def evaluate_material(board):
    """Return material from the side-to-move's perspective."""
    white_score = 0
    black_score = 0
    for piece in range(5):
        white_score += PIECE_VALUES[piece] * popcount(board[piece])
        black_score += PIECE_VALUES[piece] * popcount(board[piece + 6])
    score = white_score - black_score
    return score if board[bb_numba.STM] == 0 else -score


@njit(cache=False)
def _negamax(board, depth, alpha, beta, ply, node_limit, state):
    if state[STATE_NODES] >= node_limit:
        state[STATE_ABORTED] = 1
        return 0
    state[STATE_NODES] += 1

    if depth <= 0 or ply >= MAX_PLY:
        return evaluate_material(board)

    moves = np.empty(256, dtype=np.int32)
    move_count = bb_numba.gen_legal(board, moves)
    if move_count == 0:
        king = board[bb_numba.WK] if board[bb_numba.STM] == 0 else board[bb_numba.BK]
        king_square = bb_numba.bsf(king)
        in_check = bb_numba.is_attacked(board, king_square, board[bb_numba.STM] != 0)
        return -MATE_SCORE + ply if in_check else 0

    best_score = -INFINITY
    for index in range(move_count):
        child = bb_numba.make(board, moves[index])
        score = -_negamax(child, depth - 1, -beta, -alpha, ply + 1, node_limit, state)
        if state[STATE_ABORTED]:
            return 0
        if score > best_score:
            best_score = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best_score


@njit(cache=False)
def search_root(board, depth, node_limit, state):
    """Search one bounded iteration and return move, score, and completion."""
    state[STATE_NODES] = 0
    state[STATE_ABORTED] = 0
    if depth < 1:
        depth = 1
    if depth > MAX_SEARCH_DEPTH:
        depth = MAX_SEARCH_DEPTH
    if node_limit < 1:
        state[STATE_ABORTED] = 1
        return -1, 0, False

    moves = np.empty(256, dtype=np.int32)
    move_count = bb_numba.gen_legal(board, moves)
    if move_count == 0:
        return -1, 0, True

    best_move = moves[0]
    best_score = -INFINITY
    alpha = -INFINITY
    beta = INFINITY
    for index in range(move_count):
        if state[STATE_NODES] >= node_limit:
            state[STATE_ABORTED] = 1
            return best_move, best_score, False
        child = bb_numba.make(board, moves[index])
        score = -_negamax(child, depth - 1, -beta, -alpha, 1, node_limit, state)
        if state[STATE_ABORTED]:
            return best_move, best_score, False
        if score > best_score:
            best_score = score
            best_move = moves[index]
        if score > alpha:
            alpha = score
    return best_move, best_score, True


def new_state() -> np.ndarray:
    """Create explicit mutable state for one search iteration."""
    return np.zeros(2, dtype=np.int64)


def _warmup() -> None:
    board = bb_numba.board_np(START_FEN)
    search_root(board, 1, 1_000, new_state())


_warmup()
