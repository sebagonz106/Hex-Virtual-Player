"""Basic MCTS player for Hex game."""

from __future__ import annotations
import time
import random
import math
from typing import Optional, Tuple

from board import HexBoard
from players.player import Player
from players.utils.board_optimized import BoardOptimized
from players.utils.early_check import (
    get_immediate_winning_move,
    get_opponent_forcing_move,
    suggest_opening_move,
)


class _MCTSRecyclerNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        board: BoardOptimized,
        player_id: int,
        parent: Optional[_MCTSRecyclerNode] = None,
        depth: int = 0,
    ):
        """
        Initialize MCTS node.
        
        Args:
            board: Optimized board state.
            player_id: ID of player to move (1 or 2).
            parent: Parent node.
            depth: Depth of this node in the tree.
        """
        self.board = board
        self.player_id = player_id
        self.parent = parent
        self.depth = depth
        self.children = {}
        self.visit_count = 0
        self.win_count = 0
        self.untried_moves = list(board.get_empty_positions())
        random.shuffle(self.untried_moves)
        self.depth = depth


    def uct_value(self, exploration_c: float) -> float:
        """
        Calculate UCT value for this node.
        
        UCT = Q(v)/N(v) + C * sqrt(ln(N(parent))/N(v))
        
        Args:
            exploration_c: Exploration constant.
            
        Returns:
            UCT score.
        """
        if self.visit_count == 0:
            return float('inf')

        exploitation = 1 - self.win_count / self.visit_count
        exploration = exploration_c * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        ) if self.parent else 0
        return exploitation + exploration

    def select_best_child(self, exploration_c: float) -> Optional[_MCTSRecyclerNode]:
        """
        Select child with highest UCT value.
        
        Args:
            exploration_c: Exploration constant.
            
        Returns:
            Best child or None.
        """
        if not self.children:
            return None

        return max(self.children.values(), key=lambda child: child.uct_value(exploration_c))

    def expand(self, move: Tuple[int, int], player_id: int) -> _MCTSRecyclerNode:
        """
        Expand a new child node.
        
        Args:
            move: Move coordinates (row, col).
            player_id: Player ID for new state.
            
        Returns:
            New child node.
        """
        new_board = self.board.clone()
        new_board.place_piece(move[0], move[1], self.player_id)
        child = _MCTSRecyclerNode(new_board, player_id, parent=self, depth=self.depth + 1)
        self.children[move] = child
        return child

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return (
            self.board.is_full()
            or self.board.check_connection(1)
            or self.board.check_connection(2)
        )

    def get_winner(self) -> Optional[int]:
        """Get winner if terminal, None otherwise."""
        if self.board.check_connection(1):
            return 1
        if self.board.check_connection(2):
            return 2
        return None


class RecyclerMCTSPlayer(Player):
    """
    Monte Carlo Tree Search player for Hex with tree recycling between moves.
    
    Implements MCTS with UCT selection, random playouts, and intelligent tree reuse.
    Between moves, attempts to detect opponent's move and continue searching from
    the appropriate subtree instead of restarting from scratch.
    """

    MAX_DEPTH = 50  # Limit depth to prevent infinite loops in large boards

    def __init__(self, player_id: int, max_time: float = 4.98, exploration_c: float = 0.2):
        """
        Initialize MCTS player with tree recycling capabilities.
        
        Args:
            player_id: Player ID (1 or 2).
            max_time: Maximum time per move in seconds (default 4.999).
            exploration_c: UCT exploration constant (default 0.2).
        """
        super().__init__(player_id)
        self.max_time = max_time
        self.exploration_c = exploration_c
        
        # Tree recycling state
        self.board: Optional[BoardOptimized] = None
        self.root: Optional[_MCTSRecyclerNode] = None
        self.tree_reused_count = 0  # Statistics

    def play(self, board: HexBoard) -> Tuple[int, int]:
        """
        Select best move using MCTS with tree recycling and time control.
        
        Attempts to reuse the search tree from the previous turn by detecting
        the opponent's move and continuing from the appropriate subtree.
        
        Args:
            board: Current HexBoard state.
            
        Returns:
            Best move (row, col).
        """
        time_start = time.time()

        # Convert to optimized board immediately for all subsequent operations
        new_board = BoardOptimized(board)

        best_move = random.choice(list(new_board.get_empty_positions()))  # Fallback random move

        # Early check: opening move suggestion
        opening_move = suggest_opening_move(new_board, self.player_id)
        if opening_move and new_board.board[opening_move[0]][opening_move[1]] == 0:
            self._reset_info()
            return opening_move

        # Attempt tree recycling
        root = self._find_reusable_root(new_board)
        if root is None:
            # No recycling possible, create fresh root
            root = _MCTSRecyclerNode(new_board, self.player_id)

        # Early check: immediate winning move
        win_move = get_immediate_winning_move(new_board, self.player_id)
        if win_move:
            self._reset_info()
            return win_move

        # Early check: must block opponent
        opponent_id = 3 - self.player_id
        block_move = get_opponent_forcing_move(new_board, opponent_id)
        if block_move:
            self._save_state_for_recycling(new_board, root, block_move)
            return block_move

        # MCTS search using optimized board
        while time.time() - time_start < self.max_time:
            self._mcts_iteration(root)

        # Select best move
        best_move = self._select_best_move(root)

        # Save state for next turn's recycling
        self._save_state_for_recycling(new_board, root, best_move)

        print(self.player_id, f"- {(time.time() - time_start):.4f}, seconds for MCTS search with exploration_c =", self.exploration_c)
        
        return best_move

    def _mcts_iteration(self, root: _MCTSRecyclerNode) -> None:
        """
        Execute one MCTS iteration: Selection -> Expansion -> Simulation -> Backpropagation.
        
        Args:
            root: Root node of MCTS tree.
        """
        # Selection + Expansion
        node = root

        while not node.is_terminal() and node.depth <  self.MAX_DEPTH:
            if node.untried_moves:
                # Expansion: try a random untried move
                move = node.untried_moves.pop()
                next_player_id = 3 - node.player_id
                node = node.expand(move, next_player_id)
                break
            else:
                # Selection: use UCT to select best child
                child = node.select_best_child(self.exploration_c)
                if child is None:
                    break
                node = child

        # Simulation: random playout from node
        if node.is_terminal():
            result = node.get_winner()
        else:
            result = self._play_random_playout(node)

        # Backpropagation: update all nodes in path
        while node is not None:
            node.visit_count += 1
            if result == node.player_id:
                node.win_count += 1
            node = node.parent

    def _play_random_playout(self, node: _MCTSRecyclerNode) -> int:
        """
        Play a random game from node until terminal.
        
        Args:
            node: Starting node.
            
        Returns:
            Winner (1 or 2).
        """
        board = node.board.clone()
        current_player = node.player_id

        # Play until board is full or someone wins
        empty_positions = list(board.get_empty_positions())
        random.shuffle(empty_positions)

        for r, c in empty_positions:
            board.place_piece(r, c, current_player)

            if board.check_connection(current_player):
                return current_player

            current_player = 3 - current_player

        # Should not reach here in Hex (no draws) but return opponent if board full
        return 3 - current_player

    def _find_board_difference(self, board1: BoardOptimized, board2: BoardOptimized) -> Optional[Tuple[int, int]]:
        """
        Find the single move that differs between two boards.
        
        Detects what move was made by comparing board states. If there is exactly
        one difference (empty → piece), returns that move. Otherwise returns None
        to indicate the boards are incomparable.
        
        Args:
            board1: Previous board state.
            board2: Current board state.
            
        Returns:
            The move (row, col) that differs, or None if incomparable.
        """
        if board1.size != board2.size:
            return None
        
        differences = []
        for r in range(board1.size):
            for c in range(board1.size):
                if board1.board[r][c] != board2.board[r][c]:
                    # Valid change: empty → piece
                    if board1.board[r][c] == 0 and board2.board[r][c] != 0:
                        differences.append((r, c))
                    else:
                        # Invalid change (overwrite, undo, etc)
                        return None
        
        # Accept only exactly 1 move
        if len(differences) == 1:
            return differences[0]
        else:
            return None

    def _find_reusable_root(self, current_board: BoardOptimized) -> Optional[_MCTSRecyclerNode]:
        """
        Attempt to find a reusable root node from the previous search tree.
        
        Algorithm:
        1. If no previous state, return None (no recycling)
        2. Detect opponent's move by comparing boards
        3. If move found and exists in children, return that child
        4. Otherwise return None (reset required)
        
        Args:
            current_board: Current board state after opponent's move.
            
        Returns:
            Reusable root node, or None if recycling not possible.
        """
        # Check if we have previous state to recycle from
        if self.board is None or self.root is None:
            return None
        
        # Find opponent's move by comparing boards
        opp_move = self._find_board_difference(self.board, current_board)
        
        if opp_move is None:
            # Board states are incomparable (multiple changes, invalid moves, etc)
            return None
        
        # Search for this move in the children of last root
        if opp_move in self.root.children:
            reused_node = self.root.children[opp_move]
            self.tree_reused_count += 1
            print(f"✓ Tree recycled from move {opp_move} with {reused_node.visit_count} visits")
            return reused_node
        else:
            # Move not found in tree (shouldn't happen, but fallback to reset)
            return None

    def _save_state_for_recycling(self, board: BoardOptimized, root: _MCTSRecyclerNode, best_move: Tuple[int, int]) -> None:
        """
        Save the current state for potential recycling in the next turn.
        
        Updates the board with the best move and saves both board and root.
        Next turn will attempt to detect opponent's move relative to this state.
        
        Args:
            board: Current board state (before best_move).
            root: Current root node after search.
            best_move: The move selected to play.
        """
        # Clone and update board with our move
        saved_board = board.clone()
        saved_board.place_piece(best_move[0], best_move[1], self.player_id)
        
        # Save for next turn
        self.board = saved_board
        self.root = root.children[best_move]
    
    def _reset_info(self):
        """Reset saved state info."""
        self.board = None
        self.root = None
        self.tree_reused_count = 0
        
    def _select_best_move(self, root: _MCTSRecyclerNode) -> Tuple[int, int]:
        """
        Select best move from root by highest visit count.
        
        Args:
            root: Root node.
            
        Returns:
            Best move (row, col).
        """
        if not root.children:
            # Fallback: random empty move
            empty = list(root.board.get_empty_positions())
            return random.choice(empty) if empty else (0, 0)

        total_sims = sum(child.visit_count for child in root.children.values())

        best_child = max(root.children.values(), key=lambda child: 1 - child.win_count/child.visit_count + child.visit_count/total_sims)

        # best_child = max(root.children.values(), key=lambda child: child.visit_count)
        
        print(self.player_id, "-", f"Best move has {best_child.visit_count} visits and win rate {1 - best_child.win_count/best_child.visit_count:.2f} after {total_sims} simulations")

        # Find move that led to this child
        for move, child in root.children.items():
            if child is best_child:
                return move

        return (0, 0)
