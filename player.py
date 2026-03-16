from __future__ import annotations
from board import HexBoard

class Player:

    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")
