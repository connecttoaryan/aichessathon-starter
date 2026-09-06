"""Bounded material search for the experimental Numba agent.

All mutable search state, including the fixed-size transposition table, is supplied
explicitly by the caller. No jitted function mutates module-global state.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import bb_numba

INFINITY = 1_000_000
MATE_SCORE = 100_000
MATE_TT_THRESHOLD = MATE_SCORE - 1_000
MAX_SEARCH_DEPTH = 4
MAX_PLY = 64
MAX_QUIESCENCE_PLY = 8
DEFAULT_NODE_LIMIT = 250_000
TT_SIZE = 2_048

STATE_NODES = 0
STATE_ABORTED = 1

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
NO_MOVE = -1

PIECE_VALUES = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FNV_OFFSET = np.uint64(0xCBF29CE484222325)
FNV_PRIME = np.uint64(0x100000001B3)


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
def _in_check(board):
    white = board[bb_numba.STM] == 0
    king = board[bb_numba.WK] if white else board[bb_numba.BK]
    return bb_numba.is_attacked(board, bb_numba.bsf(king), not white)


@njit(cache=False)
def _is_capture(board, move):
    if ((move >> 15) & 3) == 2:
        return True
    target = (move >> 6) & 63
    opponent_base = 6 if board[bb_numba.STM] == 0 else 0
    return (bb_numba.occ_side(board, opponent_base) & (bb_numba.ONE << np.uint64(target))) != 0


@njit(cache=False)
def _order_moves(board, moves, move_count, tt_move):
    """Place a verified TT move first, captures next, and quiet moves last."""
    first_unsorted = 0
    if tt_move != NO_MOVE:
        for index in range(move_count):
            if moves[index] == tt_move:
                moves[0], moves[index] = moves[index], moves[0]
                first_unsorted = 1
                break

    next_capture = first_unsorted
    for index in range(first_unsorted, move_count):
        if _is_capture(board, moves[index]):
            moves[next_capture], moves[index] = moves[index], moves[next_capture]
            next_capture += 1


@njit(cache=False)
def _tt_index(board, table_size):
    position_hash = FNV_OFFSET
    for index in range(16):
        position_hash ^= board[index]
        position_hash *= FNV_PRIME
    return int(position_hash & np.uint64(table_size - 1))


@njit(cache=False)
def _tt_matches(board, slot, tt_boards, tt_depths):
    if tt_depths[slot] < 0:
        return False
    index = 0
    while index < 16:
        if tt_boards[slot, index] != board[index]:
            return False
        index += 1
    return True


@njit(cache=False)
def _tt_move(board, tt_boards, tt_depths, tt_moves):
    slot = _tt_index(board, tt_depths.size)
    if _tt_matches(board, slot, tt_boards, tt_depths):
        return tt_moves[slot]
    return NO_MOVE


@njit(cache=False)
def _tt_probe(board, depth, alpha, beta, tt_boards, tt_depths, tt_scores, tt_flags):
    slot = _tt_index(board, tt_depths.size)
    if not _tt_matches(board, slot, tt_boards, tt_depths) or tt_depths[slot] < depth:
        return False, 0
    score = tt_scores[slot]
    flag = tt_flags[slot]
    if flag == TT_EXACT:
        return True, score
    if flag == TT_LOWER and score >= beta:
        return True, score
    if flag == TT_UPPER and score <= alpha:
        return True, score
    return False, 0


@njit(cache=False)
def _tt_store(
    board,
    depth,
    score,
    move,
    flag,
    tt_boards,
    tt_depths,
    tt_scores,
    tt_moves,
    tt_flags,
):
    # Mate scores depend on ply, so do not store them without normalization.
    if score >= MATE_TT_THRESHOLD or score <= -MATE_TT_THRESHOLD:
        return
    slot = _tt_index(board, tt_depths.size)
    if _tt_matches(board, slot, tt_boards, tt_depths) and tt_depths[slot] > depth:
        return
    for index in range(16):
        tt_boards[slot, index] = board[index]
    tt_depths[slot] = depth
    tt_scores[slot] = score
    tt_moves[slot] = move
    tt_flags[slot] = flag


@njit(cache=False)
def _terminal_score(board, ply):
    return -MATE_SCORE + ply if _in_check(board) else 0


@njit(cache=False)
def _quiescence(board, alpha, beta, ply, qply, node_limit, state):
    if state[STATE_NODES] >= node_limit:
        state[STATE_ABORTED] = 1
        return 0
    state[STATE_NODES] += 1

    moves = np.empty(256, dtype=np.int32)
    move_count = bb_numba.gen_legal(board, moves)
    if move_count == 0:
        return _terminal_score(board, ply)

    stand_pat = evaluate_material(board)
    if ply >= MAX_PLY or qply >= MAX_QUIESCENCE_PLY:
        return stand_pat

    in_check = _in_check(board)
    if not in_check:
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    _order_moves(board, moves, move_count, NO_MOVE)
    for index in range(move_count):
        if not in_check and not _is_capture(board, moves[index]):
            continue
        child = bb_numba.make(board, moves[index])
        score = -_quiescence(
            child,
            -beta,
            -alpha,
            ply + 1,
            qply + 1,
            node_limit,
            state,
        )
        if state[STATE_ABORTED]:
            return 0
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def _negamax(
    board,
    depth,
    alpha,
    beta,
    ply,
    node_limit,
    state,
    tt_boards,
    tt_depths,
    tt_scores,
    tt_moves,
    tt_flags,
):
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, 0, node_limit, state)
    if state[STATE_NODES] >= node_limit:
        state[STATE_ABORTED] = 1
        return 0
    state[STATE_NODES] += 1
    if ply >= MAX_PLY:
        return evaluate_material(board)

    original_alpha = alpha
    hit, tt_score = _tt_probe(
        board,
        depth,
        alpha,
        beta,
        tt_boards,
        tt_depths,
        tt_scores,
        tt_flags,
    )
    if hit:
        return tt_score

    moves = np.empty(256, dtype=np.int32)
    move_count = bb_numba.gen_legal(board, moves)
    if move_count == 0:
        return _terminal_score(board, ply)

    tt_move = _tt_move(board, tt_boards, tt_depths, tt_moves)
    _order_moves(board, moves, move_count, tt_move)
    best_move = moves[0]
    best_score = -INFINITY
    for index in range(move_count):
        child = bb_numba.make(board, moves[index])
        score = -_negamax(
            child,
            depth - 1,
            -beta,
            -alpha,
            ply + 1,
            node_limit,
            state,
            tt_boards,
            tt_depths,
            tt_scores,
            tt_moves,
            tt_flags,
        )
        if state[STATE_ABORTED]:
            return 0
        if score > best_score:
            best_score = score
            best_move = moves[index]
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    flag = TT_EXACT
    if best_score <= original_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    _tt_store(
        board,
        depth,
        best_score,
        best_move,
        flag,
        tt_boards,
        tt_depths,
        tt_scores,
        tt_moves,
        tt_flags,
    )
    return best_score


@njit(cache=False)
def search_root(
    board,
    depth,
    node_limit,
    state,
    tt_boards,
    tt_depths,
    tt_scores,
    tt_moves,
    tt_flags,
):
    """Search one bounded iteration and return move, score, and completion."""
    state[STATE_NODES] = 0
    state[STATE_ABORTED] = 0
    if depth < 1:
        depth = 1
    if depth > MAX_SEARCH_DEPTH:
        depth = MAX_SEARCH_DEPTH
    if node_limit < 1:
        state[STATE_ABORTED] = 1
        return NO_MOVE, 0, False

    moves = np.empty(256, dtype=np.int32)
    move_count = bb_numba.gen_legal(board, moves)
    if move_count == 0:
        return NO_MOVE, 0, True

    tt_move = _tt_move(board, tt_boards, tt_depths, tt_moves)
    _order_moves(board, moves, move_count, tt_move)
    best_move = moves[0]
    best_score = -INFINITY
    alpha = -INFINITY
    beta = INFINITY
    for index in range(move_count):
        if state[STATE_NODES] >= node_limit:
            state[STATE_ABORTED] = 1
            return best_move, best_score, False
        child = bb_numba.make(board, moves[index])
        score = -_negamax(
            child,
            depth - 1,
            -beta,
            -alpha,
            1,
            node_limit,
            state,
            tt_boards,
            tt_depths,
            tt_scores,
            tt_moves,
            tt_flags,
        )
        if state[STATE_ABORTED]:
            return best_move, best_score, False
        if score > best_score:
            best_score = score
            best_move = moves[index]
        if score > alpha:
            alpha = score

    _tt_store(
        board,
        depth,
        best_score,
        best_move,
        TT_EXACT,
        tt_boards,
        tt_depths,
        tt_scores,
        tt_moves,
        tt_flags,
    )
    return best_move, best_score, True


def new_state() -> np.ndarray:
    """Create explicit mutable state for one search iteration."""
    return np.zeros(2, dtype=np.int64)


def new_transposition_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a bounded direct-mapped table with exact board verification."""
    return (
        np.zeros((TT_SIZE, 16), dtype=np.uint64),
        np.full(TT_SIZE, -1, dtype=np.int16),
        np.zeros(TT_SIZE, dtype=np.int32),
        np.full(TT_SIZE, NO_MOVE, dtype=np.int32),
        np.zeros(TT_SIZE, dtype=np.int8),
    )


def _warmup() -> None:
    board = bb_numba.board_np(START_FEN)
    table = new_transposition_table()
    search_root(board, 1, 1_000, new_state(), *table)


_warmup()
