"""Optimized board wrapper with Union-Find for efficient MCTS simulations."""

from __future__ import annotations
from typing import Set, Tuple, Optional, List
from board import HexBoard


class UnionSnapshot:
    """
    Represents a snapshot of a Union-Find node state before a union operation.
    
    Stores the previous parent and rank values to enable reverting union operations.
    """
    
    def __init__(self, node_idx: int, prev_parent: int, prev_rank: int):
        """
        Args:
            node_idx: Index of the node whose state is captured.
            prev_parent: Previous value of parent[node_idx].
            prev_rank: Previous value of rank[node_idx].
        """
        self.node_idx = node_idx
        self.prev_parent = prev_parent
        self.prev_rank = prev_rank


class BoardOptimized:
    """
    Optimized board representation with Union-Find for fast connection checking.
    """

    def __init__(self, hex_board: HexBoard):
        """
        Initialize from HexBoard with Union-Find structure.
        
        Args:
            hex_board: HexBoard instance to wrap.
            
        Raises:
            ValueError: If board size is invalid.
        """
        if not (0 < hex_board.size <= 25):
            raise ValueError(f"Invalid board size: {hex_board.size}")

        self.size = hex_board.size
        self.board = [row[:] for row in hex_board.board]
        self._empty_positions: Set[Tuple[int, int]] = self._compute_empty_positions()
        self._neighbors_cache: dict = {}

        # Union-Find structures (N² cells + 4 phantom nodes)
        total_nodes = self.size * self.size + 4
        self.parent = list(range(total_nodes))
        self.rank = [0] * total_nodes

        # Phantom nodes for win detection:
        # - Player 1 (horizontal): connects left and right borders
        # - Player 2 (vertical): connects top and bottom borders
        self.P1_LEFT = self.size * self.size
        self.P1_RIGHT = self.size * self.size + 1
        self.P2_TOP = self.size * self.size + 2
        self.P2_BOTTOM = self.size * self.size + 3

        # Move history stack: each entry is a list of UnionSnapshot objects
        # Enables undo operations without full state duplication
        self.move_union_stack: List[List[UnionSnapshot]] = []
        self.move_history: List[Tuple[int, int]] = []

        self._initialize_union_find()

    def _compute_empty_positions(self) -> Set[Tuple[int, int]]:
        """Compute set of all empty positions."""
        empty = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    empty.add((r, c))
        return empty

    def _cell_to_idx(self, r: int, c: int) -> int:
        """Convert (row, col) to linear index."""
        return r * self.size + c

    def _idx_to_cell(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (row, col)."""
        return divmod(idx, self.size)

    def _find(self, x: int) -> int:
        """
        Find the root of element x in the Union-Find structure.
        """
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def _union(self, x: int, y: int) -> Optional[UnionSnapshot]:
        """
        Perform union of two sets using union by rank strategy.
        
        Args:
            x: First element.
            y: Second element.
            
        Returns:
            UnionSnapshot if union was performed, None if already in same component.
        """
        x_root = self._find(x)
        y_root = self._find(y)

        if x_root == y_root:
            return None  # Already in same component

        # Perform union by rank
        if self.rank[x_root] < self.rank[y_root]:
            snapshot = UnionSnapshot(x_root, self.parent[x_root], self.rank[x_root])
            self.parent[x_root] = y_root
        else:
            snapshot = UnionSnapshot(y_root, self.parent[y_root], self.rank[y_root])
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

        return snapshot

    def _initialize_union_find(self) -> None:
        """
        Initialize the Union-Find structure from the current board state.
        
        Connects adjacent pieces of the same player and border pieces to phantom nodes.
        """
        # Connect adjacent pieces of the same player
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] != 0:
                    player_id = self.board[r][c]
                    idx = self._cell_to_idx(r, c)

                    # Process all neighbors; redundant unions return None
                    for nr, nc in self._neighbors(r, c):
                        if self.board[nr][nc] == player_id:
                            n_idx = self._cell_to_idx(nr, nc)
                            self._union(idx, n_idx)

        # Connect border pieces to phantom nodes
        for r in range(self.size):
            if self.board[r][0] == 1:
                self._union(self._cell_to_idx(r, 0), self.P1_LEFT)
            if self.board[r][self.size - 1] == 1:
                self._union(self._cell_to_idx(r, self.size - 1), self.P1_RIGHT)

        for c in range(self.size):
            if self.board[0][c] == 2:
                self._union(self._cell_to_idx(0, c), self.P2_TOP)
            if self.board[self.size - 1][c] == 2:
                self._union(self._cell_to_idx(self.size - 1, c), self.P2_BOTTOM)

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """
        Place a piece on the board and update the Union-Find structure.
        
        All unions performed are recorded atomically for reliable undo operations.
        
        Args:
            row: Row coordinate.
            col: Column coordinate.
            player_id: Player ID (1 or 2).
            
        Returns:
            True if successful, False otherwise.
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row][col] != 0:
            return False

        self.board[row][col] = player_id
        self._empty_positions.discard((row, col))

        idx = self._cell_to_idx(row, col)
        unions_in_move: List[UnionSnapshot] = []

        # Connect to phantom nodes if on board edge
        if player_id == 1:
            if col == 0:
                snapshot = self._union(idx, self.P1_LEFT)
                if snapshot:
                    unions_in_move.append(snapshot)
            if col == self.size - 1:
                snapshot = self._union(idx, self.P1_RIGHT)
                if snapshot:
                    unions_in_move.append(snapshot)
        else:  # player_id == 2
            if row == 0:
                snapshot = self._union(idx, self.P2_TOP)
                if snapshot:
                    unions_in_move.append(snapshot)
            if row == self.size - 1:
                snapshot = self._union(idx, self.P2_BOTTOM)
                if snapshot:
                    unions_in_move.append(snapshot)

        # Connect with neighbors of the same color
        for nr, nc in self._neighbors(row, col):
            if self.board[nr][nc] == player_id:
                n_idx = self._cell_to_idx(nr, nc)
                snapshot = self._union(idx, n_idx)
                if snapshot:
                    unions_in_move.append(snapshot)

        # Record all unions from this move for atomic undo
        self.move_union_stack.append(unions_in_move)
        self.move_history.append((row, col))

        return True

    def undo_move(self) -> bool:
        """
        Revert the last move and all associated unions.
        
        Returns:
            True if successful, False if no moves to undo.
        """
        if not self.move_history:
            return False

        row, col = self.move_history.pop()
        self.board[row][col] = 0
        self._empty_positions.add((row, col))

        # Retrieve all unions from the last move
        unions_to_undo = self.move_union_stack.pop()

        # Revert each union in reverse order
        for snapshot in reversed(unions_to_undo):
            self.parent[snapshot.node_idx] = snapshot.prev_parent
            self.rank[snapshot.node_idx] = snapshot.prev_rank

        return True

    def check_connection(self, player_id: int) -> bool:
        """
        Check if the player has achieved a winning connection.
        
        Args:
            player_id: Player ID (1 or 2).
            
        Returns:
            True if player has a winning connection, False otherwise.
        """
        if player_id == 1:
            # Player 1 wins if left and right borders are connected
            return self._find(self.P1_LEFT) == self._find(self.P1_RIGHT)
        else:
            # Player 2 wins if top and bottom borders are connected
            return self._find(self.P2_TOP) == self._find(self.P2_BOTTOM)

    def get_empty_positions(self) -> Set[Tuple[int, int]]:
        """Get current empty positions."""
        return self._empty_positions.copy()

    def _neighbors(self, row: int, col: int) -> list:
        """
        Get adjacent neighbors in even-r hexagonal grid layout.
        
        Results are cached for performance optimization.
        
        Args:
            row: Row coordinate.
            col: Column coordinate.
            
        Returns:
            List of adjacent neighbor coordinates as (row, col) tuples.
        """
        key = (row, col)
        if key in self._neighbors_cache:
            return self._neighbors_cache[key]

        neighbors = []
        if row % 2 == 0:
            directions = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))

        self._neighbors_cache[key] = neighbors
        return neighbors

    def is_full(self) -> bool:
        """Check if board is completely filled."""
        return len(self._empty_positions) == 0

    def is_empty(self) -> bool:
        """Check if board is completely empty."""
        return len(self._empty_positions) == self.size * self.size

    def count_pieces(self, player_id: int) -> int:
        """Count pieces for a given player."""
        count = 0
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == player_id:
                    count += 1
        return count

    def total_pieces(self) -> int:
        """Count total non-empty pieces."""
        return self.size * self.size - len(self._empty_positions)

    def clone(self) -> "BoardOptimized":
        """
        Create a deep copy of this board with complete state.
        
        Includes Union-Find structure and move history.
        """
        new_board = BoardOptimized.__new__(BoardOptimized)
        new_board.size = self.size
        new_board.board = [row[:] for row in self.board]
        new_board._empty_positions = self._empty_positions.copy()
        new_board._neighbors_cache = self._neighbors_cache

        # Union-Find state
        new_board.parent = self.parent.copy()
        new_board.rank = self.rank.copy()

        # Phantom nodes
        new_board.P1_LEFT = self.P1_LEFT
        new_board.P1_RIGHT = self.P1_RIGHT
        new_board.P2_TOP = self.P2_TOP
        new_board.P2_BOTTOM = self.P2_BOTTOM

        # Move history
        new_board.move_union_stack = [
            [UnionSnapshot(s.node_idx, s.prev_parent, s.prev_rank) for s in move]
            for move in self.move_union_stack
        ]
        new_board.move_history = self.move_history.copy()

        return new_board
