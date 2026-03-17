"""Early move detection for immediate tactical gains."""

from __future__ import annotations
from typing import Optional, Tuple
from players.utils.board_optimized import BoardOptimized


def get_immediate_winning_move(board: BoardOptimized, player_id: int) -> Optional[Tuple[int, int]]:
    """
    Detect a move that wins immediately.
    
    Complexity: O(K * N^2) where K ~ 5-20 candidate moves near own pieces.
    
    Args:
        board: BoardOptimized instance.
        player_id: Player ID (1 or 2).
        
    Returns:
        Winning move (row, col) or None.
    """
    # Find candidate moves: empty cells adjacent to own pieces
    candidates = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.board[r][c] == player_id:
                for nr, nc in board._neighbors(r, c):
                    if board.board[nr][nc] == 0:
                        candidates.add((nr, nc))

    # Test each candidate
    for r, c in candidates:
        try:
            if board.place_piece(r, c, player_id):
                if board.check_connection(player_id):
                    return (r, c)
        finally:
            board.undo_move()  # Undo the test move

    return None


def get_opponent_forcing_move(board: BoardOptimized, opp_id: int) -> Optional[Tuple[int, int]]:
    """
    Detect if opponent has a winning move. Must block it.
    
    Args:
        board: BoardOptimized instance.
        opp_id: Opponent's player ID (1 or 2).
        
    Returns:
        Opponent's winning move to block (row, col) or None.
    """
    return get_immediate_winning_move(board, opp_id)


def suggest_opening_move(board: BoardOptimized, player_id: int) -> Optional[Tuple[int, int]]:
    """
    Suggest a strong opening move for early game.
    
    Strategy:
    - If center is empty: play center
    - If center occupied with 1 total piece: play neighbor closest to objective
    
    Args:
        board: BoardOptimized instance.
        player_id: Player ID (1 or 2).
        
    Returns:
        Suggested opening move (row, col) or None.
    """
    center_r, center_c = board.size // 2, board.size // 2

    # If center is empty, play it
    if board.board[center_r][center_c] == 0:
        return (center_r, center_c)

    # If only 1 piece on board and center is occupied, play strategic neighbor
    total_pieces = board.total_pieces()

    if total_pieces == 1:
        # Get neighbors of center
        neighbors = list(board._neighbors(center_r, center_c))
        empty_neighbors = [
            (r, c) for r, c in neighbors if board.board[r][c] == 0
        ]

        if not empty_neighbors:
            return None

        # Choose neighbor closest to objective edge
        if player_id == 1:
            # Player 1 wants to be close to right edge (col = size - 1)
            best = max(empty_neighbors, key=lambda pos: pos[1])
        else:
            # Player 2 wants to be close to bottom edge (row = size - 1)
            best = max(empty_neighbors, key=lambda pos: pos[0])

        return best

    return None
