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


class _MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        board: BoardOptimized,
        player_id: int,
        parent: Optional[_MCTSNode] = None,
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

    def select_best_child(self, exploration_c: float) -> Optional[_MCTSNode]:
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

    def expand(self, move: Tuple[int, int], player_id: int) -> _MCTSNode:
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
        child = _MCTSNode(new_board, player_id, parent=self, depth=self.depth + 1)
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


class BasicMCTSPlayer(Player):
    """
    Monte Carlo Tree Search player for Hex.
    
    Implements MCTS with UCT selection and random playouts.
    """

    MAX_DEPTH = 50  # Limit depth to prevent infinite loops in large boards

    def __init__(self, player_id: int, max_time: float = 4.999, exploration_c: float = 0.1):
        """
        Initialize MCTS player.
        
        Args:
            player_id: Player ID (1 or 2).
            max_time: Maximum time per move in seconds (default 4.999).
            exploration_c: UCT exploration constant (default 0.1).
        """
        super().__init__(player_id)
        self.max_time = max_time
        self.exploration_c = exploration_c

    def play(self, board: HexBoard) -> Tuple[int, int]:
        """
        Select best move using MCTS with time control.
        
        Args:
            board: Current HexBoard state.
            
        Returns:
            Best move (row, col).
        """
        time_start = time.time()

        # Convert to optimized board immediately for all subsequent operations
        opt_board = BoardOptimized(board)

        # Early check: immediate winning move
        win_move = get_immediate_winning_move(opt_board, self.player_id)
        if win_move:
            return win_move

        # Early check: must block opponent
        opponent_id = 3 - self.player_id
        block_move = get_opponent_forcing_move(opt_board, opponent_id)
        if block_move:
            return block_move

        # Early check: opening move suggestion
        opening_move = suggest_opening_move(opt_board, self.player_id)
        if opening_move and opt_board.board[opening_move[0]][opening_move[1]] == 0:
            return opening_move
        
        best_move = random.choice(list(opt_board.get_empty_positions()))  # Fallback random move

        # MCTS search using optimized board
        # Root node represents current board state with MCTS player as next to move
        root = _MCTSNode(opt_board, self.player_id)

        while time.time() - time_start < self.max_time:
            self._mcts_iteration(root)

        # Select best move by visit count
        best_move = self._select_best_move(root)

        print(time.time() - time_start, "seconds for MCTS search with exploration_c =", self.exploration_c)
        
        return best_move

    def _mcts_iteration(self, root: _MCTSNode) -> None:
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

    def _play_random_playout(self, node: _MCTSNode) -> int:
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

    def _select_best_move(self, root: _MCTSNode) -> Tuple[int, int]:
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
        
        print(f"Best move has {best_child.visit_count} visits and win rate {1 - best_child.win_count/best_child.visit_count:.2f} after {total_sims} simulations")

        # Find move that led to this child
        for move, child in root.children.items():
            if child is best_child:
                return move

        return (0, 0)
