from __future__ import annotations

from collections import deque


class HexBoard:
    """
    Tablero de HEX usando una matriz NxN y vecindad even-r layout.

    Convención:
      0 = vacío
      1 = jugador horizontal (izquierda -> derecha)
      2 = jugador vertical (arriba -> abajo)
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size debe ser positivo")
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def clone(self) -> "HexBoard":
        new_board = HexBoard(self.size)
        new_board.board = [row[:] for row in self.board]
        return new_board

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row][col] != 0:
            return False
        
        self.board[row][col] = player_id
        return True

    def _neighbors(self, row: int, col: int):
        # even-r layout
        if row % 2 == 0:
            directions = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def check_connection(self, player_id: int) -> bool:
        """
        Jugador 1 gana conectando izquierda-derecha.
        Jugador 2 gana conectando arriba-abajo.
        """

        visited = set()
        queue = deque()

        if player_id == 1:
            for r in range(self.size):
                if self.board[r][0] == 1:
                    queue.append((r, 0))
                    visited.add((r, 0))

            while queue:
                r, c = queue.popleft()
                if c == self.size - 1:
                    return True
                for nr, nc in self._neighbors(r, c):
                    if (nr, nc) not in visited and self.board[nr][nc] == 1:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        else:
            for c in range(self.size):
                if self.board[0][c] == 2:
                    queue.append((0, c))
                    visited.add((0, c))

            while queue:
                r, c = queue.popleft()
                if r == self.size - 1:
                    return True
                for nr, nc in self._neighbors(r, c):
                    if (nr, nc) not in visited and self.board[nr][nc] == 2:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return False

    def is_full(self) -> bool:
        return all(cell != 0 for row in self.board for cell in row)

    def __str__(self) -> str:
        lines = []
        for r in range(self.size):
            indent = " " * r
            row_symbols = []
            for c in range(self.size):
                value = self.board[r][c]
                row_symbols.append("." if value == 0 else str(value))
            lines.append(indent + " ".join(row_symbols))
        return "\n".join(lines)
