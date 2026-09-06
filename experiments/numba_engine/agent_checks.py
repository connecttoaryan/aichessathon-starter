"""Focused correctness checks for the experimental playable agent."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import chess
import numpy as np

from . import agent, bb_numba, search


@dataclass(frozen=True)
class PositionCase:
    name: str
    fen: str
    required_moves: frozenset[str] = frozenset()


POSITION_CASES = (
    PositionCase("ordinary", chess.STARTING_FEN),
    PositionCase(
        "promotion",
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        frozenset({"a7a8q", "a7a8r", "a7a8b", "a7a8n"}),
    ),
    PositionCase(
        "castling",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        frozenset({"e1g1", "e1c1"}),
    ),
    PositionCase(
        "en_passant",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        frozenset({"e5d6"}),
    ),
)

MATE_CASES = (
    PositionCase("white_mate_in_one", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
    PositionCase("black_mate_in_one", "8/8/8/8/8/6k1/5q2/7K b - - 0 1"),
)


def _numba_legal_moves(fen: str) -> set[str]:
    board = bb_numba.board_np(fen)
    moves = np.empty(256, dtype=np.int32)
    count = int(bb_numba.gen_legal(board, moves))
    return {agent.move_to_uci(int(moves[index])) for index in range(count)}


def _python_chess_legal_moves(fen: str) -> set[str]:
    return {move.uci() for move in chess.Board(fen).legal_moves}


def check_special_positions() -> bool:
    passed = True
    for case in POSITION_CASES:
        expected = _python_chess_legal_moves(case.fen)
        actual = _numba_legal_moves(case.fen)
        chosen = agent.get_move(case.fen, 10_000)
        ok = actual == expected and case.required_moves <= actual and chosen in expected
        passed &= ok
        print(f"{case.name}: {'PASS' if ok else 'FAIL'} legal={len(actual)} chosen={chosen}")
        if not ok:
            print(f"  missing_vs_python_chess={sorted(expected - actual)}")
            print(f"  extra_vs_python_chess={sorted(actual - expected)}")
            print(f"  missing_required={sorted(case.required_moves - actual)}")
    return passed


def check_mates() -> bool:
    passed = True
    for case in MATE_CASES:
        board = chess.Board(case.fen)
        chosen = agent.get_move(case.fen, 10_000)
        move = chess.Move.from_uci(chosen)
        legal = move in board.legal_moves
        if legal:
            board.push(move)
        ok = legal and board.is_checkmate()
        passed &= ok
        print(f"{case.name}: {'PASS' if ok else 'FAIL'} chosen={chosen}")
    return passed


def check_node_guard() -> bool:
    board = bb_numba.board_np(chess.STARTING_FEN)
    state = search.new_state()
    _move, _score, completed = search.search_root(board, search.MAX_SEARCH_DEPTH, 1, state)
    ok = not completed and state[search.STATE_ABORTED] == 1 and state[search.STATE_NODES] <= 1
    print(
        f"node_guard: {'PASS' if ok else 'FAIL'} "
        f"nodes={state[search.STATE_NODES]} completed={completed}"
    )
    return bool(ok)


def main() -> int:
    if check_special_positions() and check_mates() and check_node_guard():
        print("Agent checks: PASS")
        return 0
    print("Agent checks: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
