"""
Monte Carlo Tree Search player for Hex with Progressive Strategies.

Implements MCTS with:
- RAVE/AMAF enhancement for faster convergence
- Tree recycling between moves for efficiency
- Phase-adaptive parameters (Progressive Bias) - Chaslot 2008
"""

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
from players.utils.game_phase import GamePhaseManager


class _ProgressiveMCTSNode:
    """Node in the MCTS tree with RAVE/AMAF statistics."""

    def __init__(
        self,
        board: BoardOptimized,
        player_id: int,
        parent: Optional[_ProgressiveMCTSNode] = None,
        depth: int = 0,
    ):
        """
        Initialize MCTS node with RAVE statistics.
        
        Args:
            board: Optimized board state.
            player_id: ID of player to move (1 or 2).
            parent: Parent node.
            depth: Depth of this node in the tree.
        """
        self.board = board
        self.player_id = player_id
        self.parent: Optional[_ProgressiveMCTSNode] = parent
        self.depth = depth
        self.children = {}
        
        # UCT statistics
        self.visit_count = 0
        self.win_count = 0
        
        # RAVE/AMAF statistics (All-Moves-As-First)
        # Maps move -> (visit_count, win_count)
        self.amaf_visits = {}  # {move: count}
        self.amaf_wins = {}    # {move: count}
        
        # Reverse mapping for O(1) lookup: child node -> move
        # Used in select_best_child_with_rave to avoid O(n) loop
        self.reverse_children = {}  # {id(child): move}
        
        self.untried_moves = list(board.get_empty_positions())
        random.shuffle(self.untried_moves)


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

    def select_best_child_with_rave(self, exploration_c: float, rave_bias: float = 0.00025) -> Optional[_ProgressiveMCTSNode]:
        """
        Select child using UCT+RAVE combined value.
        
        Implements the formula from Cazenave & Saffidine (2009) Monte-Carlo Hex:
        V_combined = (1 - coef) * UCT_value + coef * AMAF_value
        
        Where:
        coef = rc / (rc + c + rc * c * bias)
        
        And:
        - rc: amaf_visits (number of simulations where move was played)
        - rw: amaf_wins (number of wins with this move)
        - c: visit_count (simulations at this node)
        - bias: 0.00025 (empirically optimal from Table 4, Cazenave 2009)
        
        Args:
            exploration_c: Exploration constant for UCT.
            rave_bias: Bias for decay function (default 0.00025).
            
        Returns:
            Best child by RAVE-combined value, or None if no children.
        """
        if not self.children:
            return None
        
        def combined_value(child):
            # UCT value (traditional component)
            uct_val = child.uct_value(exploration_c)
            
            # Get move leading to this child using reverse mapping (O(1))
            move_to_child = self.reverse_children.get(id(child))
            
            if move_to_child is None or move_to_child not in self.amaf_visits:
                # No AMAF data yet, return pure UCT
                return uct_val
            
            # Calculate AMAF win rate
            amaf_visits_count = self.amaf_visits[move_to_child]
            if amaf_visits_count == 0:
                return uct_val
            
            amaf_win_rate = 1 - self.amaf_wins[move_to_child] / amaf_visits_count
            
            # coef = rc / (rc + c + rc ∗ c ∗ bias) decays with visit count
            rc = amaf_visits_count
            c = self.visit_count
            coef = rc / (rc + c + rc * c * rave_bias)
            
            # Combine UCT and RAVE
            combined_val = (1.0 - coef) * uct_val + coef * amaf_win_rate
            
            return combined_val
        
        return max(self.children.values(), key=combined_value)

    def expand(self, move: Tuple[int, int], player_id: int) -> _ProgressiveMCTSNode:
        """
        Expand a new child node by placing a move on the board.
        
        Args:
            move: Move coordinates (row, col).
            player_id: Player ID for new state.
            
        Returns:
            New child node created from this expansion.
        """
        
        new_board = self.board.clone()
        new_board.place_piece(move[0], move[1], self.player_id)
        child = _ProgressiveMCTSNode(new_board, player_id, parent=self, depth=self.depth + 1)
        self.children[move] = child
        self.reverse_children[id(child)] = move  # O(1) reverse lookup
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


class ProgressiveMCTSPlayer(Player):
    """
    Monte Carlo Tree Search player for Hex with RAVE/AMAF and tree recycling.
    
    Implements MCTS with:
    
    - Tree Recycling: Between moves, detects opponent's move via board
       comparison and reuses the search tree, improving efficiency.
    
    - RAVE: All-Moves-As-First statistics track value estimates
      for moves regardless of their position in the tree. Combines UCT and
      RAVE for faster convergence during early iterations.

    - Phase-Adaptive Strategies: Adjusts exploration and RAVE parameters
      dynamically based on the current game phase (OPENING, MIDGAME, ENDGAME)
      to optimize search efficiency and move quality.

    """

    MAX_DEPTH = 50  # Limit depth to prevent infinite loops in large boards
    ALPHA = 0.7  # visit count importance factor for final move selection

    def __init__(self, player_id: int, max_time: float = 4.98):
        """
        Initialize Progressive-MCTS player with Phase-Adaptive Strategies.
        
        Args:
            player_id: Player ID (1 or 2).
            max_time: Maximum time per move in seconds (default 4.98).
        """
        super().__init__(player_id)
        self.max_time = max_time
        
        # Phase-aware parameter management
        self.phase_manager = GamePhaseManager()
        
        # Tree recycling state
        self.board: Optional[BoardOptimized] = None
        self.root: Optional[_ProgressiveMCTSNode] = None
        self.tree_reused_count = 0  # Statistics: tracks successful tree reuses

    def play(self, board: HexBoard) -> Tuple[int, int]:
        """
        Select best move using MCTS with Progressive Strategies, tree recycling, and time control.
        
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
            root = _ProgressiveMCTSNode(new_board, self.player_id)

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

        # MCTS search with dynamic phase-adapted parameters
        while time.time() - time_start < self.max_time:
            self._mcts_iteration(root)

        # Select best move
        best_move = self._select_best_move(root)

        # Save state for next turn's recycling
        self._save_state_for_recycling(new_board, root, best_move)

        elapsed = time.time() - time_start
        root_phase = self._calculate_phase(root)
        params = self.phase_manager.get_parameters(root_phase)
        print(
            f"[Player {self.player_id}] Progressive-MCTS| "
            f"iterations={root.visit_count} | time={elapsed:.4f}s | tree_reused={self.tree_reused_count} times"
        )
        
        return best_move

    def _mcts_iteration(self, root: _ProgressiveMCTSNode) -> None:
        """
        Execute one MCTS iteration with RAVE/AMAF and dynamic phase-adaptive parameters.
        
        Phase is calculated per node during tree descent based on node.visit_count (iterations).
        This allows each node to independently determine its search phase maturity.
        
        Phases of one iteration:
        1. Selection: descend using UCT+RAVE with phase-specific parameters
        2. Expansion: add new node with untried move
        3. Simulation: random playout with AMAF tracking
        4. Backpropagation: update UCT and AMAF statistics along path
        
        Args:
            root: Root node of MCTS tree.
        """
        # Selection + Expansion
        node = root

        while not node.is_terminal() and node.depth < self.MAX_DEPTH:
            # Calculate phase for current node based on its visit_count (search maturity)
            current_phase = self._calculate_phase(node)
            params = self.phase_manager.get_parameters(current_phase)
            
            if node.untried_moves:
                # Expansion: try a random untried move
                move = node.untried_moves.pop()
                next_player_id = 3 - node.player_id
                
                # Expand new child node
                child = node.expand(move, next_player_id)
                node = child
                break
            else:
                # Selection: use UCT+RAVE to select best child
                child = node.select_best_child_with_rave(params["exploration_c"], params["rave_bias"])
                if child is None:
                    break
                node = child

        # Simulation: random playout from node with AMAF tracking
        if node.is_terminal():
            result = node.get_winner()
            amaf_sequence = []
        else:
            result, amaf_sequence = self._play_random_playout_with_amaf(node)

        # Backpropagation: update UCT path and AMAF statistics
        current = node
        while current is not None:
            # Traditional UCT update
            current.visit_count += 1
            if result == current.player_id:
                current.win_count += 1
            
            # All moves in playout get updated
            # AMAF statistics persist in the recycled tree
            if current.parent is not None and amaf_sequence:
                for move in amaf_sequence:
                    if move not in current.amaf_visits:
                        current.amaf_visits[move] = 0
                        current.amaf_wins[move] = 0
                    
                    current.amaf_visits[move] += 1
                    if result == current.player_id:
                        current.amaf_wins[move] += 1

            current = current.parent

    def _play_random_playout_with_amaf(self, node: _ProgressiveMCTSNode) -> Tuple[int, list]:
        """
        Play a random playout from node, tracking All-Moves-As-First (AMAF).
        
        This function differs from _play_random_playout in that it returns
        both the winner AND the sequence of moves played. This sequence is used
        to update AMAF statistics at each node in the tree.
        
        Scientific basis: Gelly & Silver (2011) "Monte-Carlo Tree Search in Hex"
        - AMAF (All-Moves-As-First) provides rapid action value estimation
        - Each move in the playout is tracked, regardless of where it was played
        - Improves convergence speed by providing early value estimates
        
        Args:
            node: Starting node for playout.
            
        Returns:
            Tuple of (winner, moves_sequence) where:
            - winner: Player ID (1 or 2)
            - moves_sequence: List of moves [(r1,c1), (r2,c2), ...] as tuples
        """
        board = node.board.clone()
        current_player = node.player_id
        moves_played = []

        # Play until board is full or someone wins
        empty_positions = list(board.get_empty_positions())
        random.shuffle(empty_positions)

        for r, c in empty_positions:
            move = (r, c)
            board.place_piece(r, c, current_player)
            moves_played.append(move)  # TRACKING: record move for AMAF

            if board.check_connection(current_player):
                return current_player, moves_played

            current_player = 3 - current_player

        # Board is full with no winner (shouldn't happen in Hex)
        return 3 - current_player, moves_played

    def _calculate_phase(self, node: _ProgressiveMCTSNode) -> str:
        """
        Determine game phase based on node visit count (iterations/search progression).
        
        Each node independently tracks its own phase based on how many simulations
        have been performed through it. Siblings or parent/child nodes can be in
        different phases simultaneously, depending on their individual visit counts.
        
        Scientific basis (Chaslot et al. 2008, "Progressive Strategies for MCTS"):
        > "A progressive strategy is similar to a simulation strategy when a few 
        > games have been played, and converges to a selection strategy when 
        > numerous games have been played."
        
        Args:
            node: MCTS node with visit_count representing iteration maturity
            
        Returns:
            Phase identifier: "OPENING" | "MIDGAME" | "ENDGAME"
        """
        return self.phase_manager.get_phase(node.visit_count) if node.parent else "MIDGAME"  # Root stays in MIDGAME phase for balanced parameters

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

    def _find_reusable_root(self, current_board: BoardOptimized) -> Optional[_ProgressiveMCTSNode]:
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
            print(f"[Player {self.player_id}] Tree recycled from move {opp_move} with {reused_node.visit_count} visits")
            reused_node.parent = None  # Detach from old parent to prevent memory leaks
            return reused_node
        else:
            # Move not found in tree (shouldn't happen, but fallback to reset)
            return None

    def _save_state_for_recycling(self, board: BoardOptimized, root: _ProgressiveMCTSNode, best_move: Tuple[int, int]) -> None:
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
        try:
            self.board = saved_board
            self.root = root.children[best_move]
            if self.root:
                if self.root.parent:
                    self.root.parent = None
        except:
            self._reset_info()
    
    def _reset_info(self):
        """Reset saved state info."""
        self.board = None
        self.root = None
        self.tree_reused_count = 0
        
    def _select_best_move(self, root: _ProgressiveMCTSNode) -> Tuple[int, int]:
        """
        Select best move from root by a heuristic approach that values both
        visit count and win rate.
        
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

        best_child = max(root.children.values(), key=lambda child: (1 - self.ALPHA) * (1 - child.win_count / child.visit_count) + self.ALPHA * child.visit_count / total_sims)

        # best_child = max(root.children.values(), key=lambda child: child.visit_count)
        
        # Safely calculate win rate (avoid division by zero)
        win_rate = (1 - best_child.win_count / best_child.visit_count) if best_child.visit_count > 0 else 0.0
        
        print(f"[Player {self.player_id}] Best move has {best_child.visit_count} visits and win rate {win_rate:.2f} after {total_sims} simulations")

        # Find move that led to this child
        for move, child in root.children.items():
            if child is best_child:
                return move

        return (0, 0)
