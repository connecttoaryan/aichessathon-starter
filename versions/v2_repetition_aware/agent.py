"""A small, readable classical chess engine for AI Chessathon."""

from __future__ import annotations

import time

import chess

PositionKey = tuple[str, str, str, str]

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Tables are written from White's point of view, starting at a1. Black's
# squares are mirrored before lookup. The modest values keep material more
# important than positional considerations.
PAWN_TABLE = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
)

KNIGHT_TABLE = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)

BISHOP_TABLE = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)

ROOK_TABLE = (
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 5, 5, 0, 0, 0,
)

QUEEN_TABLE = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)

KING_TABLE = (
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
)

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}

MATE_SCORE = 1_000_000
INFINITY = MATE_SCORE + 10_000
MAX_DEPTH = 64
MAX_QUIESCENCE_DEPTH = 8
REPETITION_AVOIDANCE_MIN_SCORE = 100
REVISIT_COST = 25
THREEFOLD_RISK_COST = 200

_deadline = 0.0
_nodes = 0
_position_counts: dict[PositionKey, int] = {}


class SearchTimeout(Exception):
    """Raised inside the tree when this move's safe search budget is spent."""


def evaluate(board: chess.Board) -> int:
    """Evaluate the position in centipawns for the side whose turn it is."""
    white_score = 0
    for square, piece in board.piece_map().items():
        table_square = square if piece.color == chess.WHITE else chess.square_mirror(square)
        value = PIECE_VALUES[piece.piece_type] + PIECE_SQUARE_TABLES[piece.piece_type][table_square]
        white_score += value if piece.color == chess.WHITE else -value
    return white_score if board.turn == chess.WHITE else -white_score


def _check_time() -> None:
    """Check time periodically, avoiding a clock call at every node."""
    global _nodes
    _nodes += 1
    if _nodes & 63 == 0 and time.perf_counter() >= _deadline:
        raise SearchTimeout


def _move_score(board: chess.Board, move: chess.Move, preferred: chess.Move | None) -> int:
    """Score a move for ordering; this score is not a chess evaluation."""
    if move == preferred:
        return 2_000_000

    score = 0
    attacker = board.piece_at(move.from_square)
    if move.promotion is not None:
        score += 900_000 + PIECE_VALUES[move.promotion]

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_value = (
            PIECE_VALUES[chess.PAWN] if victim is None else PIECE_VALUES[victim.piece_type]
        )
        attacker_value = 0 if attacker is None else PIECE_VALUES[attacker.piece_type]
        score += 500_000 + 10 * victim_value - attacker_value

    if board.gives_check(move):
        score += 100_000
    return score


def _ordered_moves(
    board: chess.Board,
    moves: list[chess.Move],
    preferred: chess.Move | None = None,
) -> list[chess.Move]:
    return sorted(moves, key=lambda move: _move_score(board, move, preferred), reverse=True)


def _is_draw(board: chess.Board) -> bool:
    """Recognise draws that can be determined from the current search position."""
    return board.is_insufficient_material() or board.halfmove_clock >= 100


def _position_key(board: chess.Board) -> PositionKey:
    """Return the parts of FEN that determine whether positions repeat."""
    placement, turn, castling, en_passant, _, _ = board.fen(en_passant="legal").split()
    return placement, turn, castling, en_passant


def _remember_position(board: chess.Board) -> None:
    """Record a position reached in the real game, never a search position."""
    key = _position_key(board)
    _position_counts[key] = _position_counts.get(key, 0) + 1


def _tactical_moves(board: chess.Board, legal_moves: list[chess.Move]) -> list[chess.Move]:
    """Return captures and promotions that can make a leaf tactically unstable."""
    return [
        move
        for move in legal_moves
        if board.is_capture(move) or move.promotion is not None
    ]


def _score_with_repetition_preference(score: int, prior_occurrences: int) -> int:
    """Prefer fresh play when ahead, especially if a move would draw immediately."""
    if prior_occurrences == 0 or score < REPETITION_AVOIDANCE_MIN_SCORE:
        return score
    if prior_occurrences >= 2:
        return score - THREEFOLD_RISK_COST
    if prior_occurrences == 1:
        return score - REVISIT_COST
    return score


def _quiescence(
    board: chess.Board,
    alpha: int,
    beta: int,
    ply: int,
    depth: int,
    legal_moves: list[chess.Move],
) -> int:
    """Continue unstable capture sequences before evaluating a leaf."""
    _check_time()

    if _is_draw(board):
        return 0

    in_check = board.is_check()
    if not in_check:
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        if depth >= MAX_QUIESCENCE_DEPTH:
            return alpha

    # All evasions must be searched in check; otherwise only tactical moves can
    # extend the leaf, keeping quiescence bounded and affordable.
    moves = legal_moves if in_check else _tactical_moves(board, legal_moves)
    for move in _ordered_moves(board, moves):
        board.push(move)
        try:
            replies = list(board.legal_moves)
            if not replies:
                score = MATE_SCORE - ply if board.is_check() else 0
            else:
                score = -_quiescence(board, -beta, -alpha, ply + 1, depth + 1, replies)
        finally:
            board.pop()

        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    """Search a position with negamax and alpha-beta pruning."""
    _check_time()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return -MATE_SCORE + ply if board.is_check() else 0
    if _is_draw(board):
        return 0
    if depth == 0:
        return _quiescence(board, alpha, beta, ply, 0, legal_moves)

    best = -INFINITY
    for move in _ordered_moves(board, legal_moves):
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1)
        finally:
            board.pop()

        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def _search_root(board: chess.Board, depth: int, preferred: chess.Move) -> tuple[int, chess.Move]:
    """Search one complete root iteration."""
    alpha = -INFINITY
    beta = INFINITY
    best_selection_score = -INFINITY
    best_raw_score = -INFINITY
    best_move = preferred

    moves = _ordered_moves(board, list(board.legal_moves), preferred)
    for move in moves:
        if time.perf_counter() >= _deadline:
            raise SearchTimeout
        board.push(move)
        try:
            prior_occurrences = _position_counts.get(_position_key(board), 0)
            score = -_negamax(board, depth - 1, -beta, -alpha, 1)
        finally:
            board.pop()

        # Repetition is a preference, never a ban. When clearly ahead, accept a
        # small evaluation cost to choose a fresh position. A materially better
        # repeating move still wins, and no penalty applies while losing.
        selection_score = _score_with_repetition_preference(score, prior_occurrences)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_raw_score = score
            best_move = move
        alpha = max(alpha, selection_score)
    return best_raw_score, best_move


def _record_choice(board: chess.Board, move: chess.Move) -> str:
    """Remember the real position produced by our move and return its UCI."""
    board.push(move)
    try:
        _remember_position(board)
    finally:
        board.pop()
    return move.uci()


def _time_budget_seconds(time_left_ms: int) -> float:
    """Spend a modest clock fraction while retaining a hard safety reserve."""
    time_left = max(0, time_left_ms) / 1_000.0
    reserve = max(0.020, min(0.500, time_left * 0.10))
    usable = max(0.0, time_left - reserve)
    target = min(2.500, time_left / 35.0)
    return min(usable, target)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move selected by time-limited iterative deepening."""
    global _deadline, _nodes

    board = chess.Board(fen)
    _remember_position(board)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"

    # Establish a legal answer before doing any timed work. Only fully completed
    # iterations replace it, so a timeout can never discard the fallback.
    best_move = legal_moves[0]
    budget = _time_budget_seconds(time_left_ms)
    if budget <= 0.0:
        return _record_choice(board, best_move)

    _deadline = time.perf_counter() + budget
    _nodes = 0
    for depth in range(1, MAX_DEPTH + 1):
        try:
            score, completed_move = _search_root(board, depth, best_move)
        except SearchTimeout:
            break
        best_move = completed_move
        if abs(score) >= MATE_SCORE - depth:
            break

    return _record_choice(board, best_move)
