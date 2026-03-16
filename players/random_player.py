from __future__ import annotations

import random
from player import Player
from board import HexBoard


def get_legal_moves(board: HexBoard):
    return [(r, c) for r in range(board.size) for c in range(board.size) if board.board[r][c] == 0]

class RandomPlayer(Player):

    def play(self, board: HexBoard) -> tuple:
        legal = get_legal_moves(board)
        if not legal:
            return (0, 0)
        return random.choice(legal)
