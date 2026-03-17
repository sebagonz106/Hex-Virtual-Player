"""Random player using optimized board representation with Union-Find."""

from __future__ import annotations

import random
from board import HexBoard
from players.player import Player
from players.utils.board_optimized import BoardOptimized
from players.utils.early_check import (
    get_immediate_winning_move,
    get_opponent_forcing_move,
    suggest_opening_move,
)


class RandomPlayer(Player):
    """
    Random player that uses BoardOptimized for efficient board operations.
    
    Selects moves randomly from available positions using the optimized
    Union-Find based board representation.

    Uses early checks to:
    - Win immediately if possible
    - Block opponent if necessary
    - Suggest optimal opening moves
    """

    def __init__(self, player_id: int):
        """
        Initialize the optimized random player.
        
        Args:
            player_id: Player ID (1 or 2).
        """
        super().__init__(player_id)
        self.board_optimized = None

    def play(self, board: HexBoard) -> tuple:
        """
        Select a random move from available positions.
        
        Args:
            board: Current HexBoard state.
            
        Returns:
            Random empty position as (row, col) tuple.
        """
        # Initialize optimized board if not yet created
        if self.board_optimized is None:
            self.board_optimized = BoardOptimized(board)
        else:
            # Sync optimized board with current state
            self.board_optimized.board = [row[:] for row in board.board]
            self.board_optimized._empty_positions = self.board_optimized._compute_empty_positions()
            self.board_optimized.move_union_stack = []
            self.board_optimized.move_history = []
            self.board_optimized._initialize_union_find()

        # Early check: immediate winning move
        win_move = get_immediate_winning_move(self.board_optimized, self.player_id)
        if win_move:
            return win_move

        # Early check: must block opponent
        opponent_id = 3 - self.player_id
        block_move = get_opponent_forcing_move(self.board_optimized, opponent_id)
        if block_move:
            return block_move

        # Early check: opening move suggestion
        opening_move = suggest_opening_move(self.board_optimized, self.player_id)
        if opening_move and self.board_optimized.board[opening_move[0]][opening_move[1]] == 0:
            return opening_move

        # Get available moves
        empty_positions = list(self.board_optimized.get_empty_positions())
        
        if not empty_positions:
            return (0, 0)
        
        return random.choice(empty_positions)
