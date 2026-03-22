"""Parallel RAVE MCTS player for Hex game with tree parallelization.

Implements tree parallelization where multiple worker threads explore a shared
MCTS tree simultaneously. Each node has fine-grained locks to synchronize
statistic updates while minimizing contention.
"""

from __future__ import annotations
import time
import random
import math
import threading
from typing import Optional, Tuple, List
from heapq import heappush, heappop
from concurrent.futures import ThreadPoolExecutor

from board import HexBoard
from players.player import Player
from players.utils.board_optimized import BoardOptimized
from players.utils.early_check import (
    get_immediate_winning_move,
    get_opponent_forcing_move,
    suggest_opening_move,
)


class _ParallelMCTSNode:
    """Node in the MCTS tree with thread-safe RAVE/AMAF statistics.
    
    Uses fine-grained locking: each node has independent locks for statistics
    updates and child expansion. This minimizes lock contention in a shared tree.
    """

    def __init__(
        self,
        board: BoardOptimized,
        player_id: int,
        parent: Optional[_ParallelMCTSNode] = None,
        depth: int = 0,
    ):
        """
        Initialize MCTS node with thread-safe RAVE statistics.
        
        Args:
            board: Optimized board state.
            player_id: ID of player to move (1 or 2).
            parent: Parent node.
            depth: Depth of this node in the tree.
        """
        self.board = board
        self.player_id = player_id
        self.parent: Optional[_ParallelMCTSNode] = parent
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
        self.reverse_children = {}  # {id(child): move}
        
        self.untried_moves = list(board.get_empty_positions())
        random.shuffle(self.untried_moves)
        
        # Thread synchronization: fine-grained locking per node
        self.stats_lock = threading.Lock()  # Protects visit_count, win_count, AMAF stats
        self.expand_lock = threading.Lock()  # Protects children dict and untried_moves


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

    def select_best_child_with_rave(self, exploration_c: float, rave_bias: float = 0.00025) -> Optional[_ParallelMCTSNode]:
        """
        Select child using UCT+RAVE combined value.
        
        Combines exploitation (UCT) with rapid value estimation (RAVE).
        Formula: V_combined = (1 - coef) * UCT + coef * RAVE
        
        Where:
        coef = rc / (rc + c + rc * c * bias)
        
        And:
        - rc: amaf_visits (number of simulations where move was played)
        - rw: amaf_wins (number of wins with this move)
        - c: visit_count (simulations at this node)
        - bias: 0.00025 (empirically optimal from Table 4, Cazenave 2009)
        
        Args:
            exploration_c: Exploration constant for UCT.
            rave_bias: Bias for decay coefficient (default 0.00025).
            
        Returns:
            Best child by combined value, or None if no children.
        """
        with self.stats_lock:
            if not self.children:
                return None
            
            def combined_value(child: _ParallelMCTSNode) -> float:
                uct_val = child.uct_value(exploration_c)
                move = self.reverse_children.get(id(child))
                
                if move is None or move not in self.amaf_visits:
                    return uct_val
                
                amaf_count = self.amaf_visits[move]
                if amaf_count == 0:
                    return uct_val
                
                amaf_rate = 1.0 - self.amaf_wins[move] / amaf_count
                rc = amaf_count
                c = self.visit_count
                coef = rc / (rc + c + rc * c * rave_bias)
                
                return (1.0 - coef) * uct_val + coef * amaf_rate
            
            empty = len(self.board.get_empty_positions())
            # Use empiric heuristics for better node selecction in advanced game simulations
            if empty < self.board.size * math.sqrt(self.board.size) / 2:
                children = {}
                children_norm = {}
                max_val = -1
                count = 0

                for child in self.children.values():
                    move = self.reverse_children.get(id(child))
                    if move is None:
                        continue
                    info = self.board.move_priority_info(self.player_id, move)
                    val = 5 * info[2] + 10 * info[3] # Favour building 2-bridges
                    children[child] = val
                    if val > 0:
                        count += 1
                    max_val = max(max_val, children[child])

                if count > 0:

                    for child, val in children.items():
                        children_norm[child] = val / (max_val * empty) # value normalization

                    max_pair = (None, -1)
                    for child in children.keys():
                        children_norm[child] += combined_value(child) # combining with rave value
                        if children_norm[child] > max_pair[1]:
                            max_pair = (child, children_norm[child])

                    return max_pair[0]

            return max(self.children.values(), key=combined_value)

    def expand_and_get_child(self, move: Tuple[int, int], player_id: int) -> _ParallelMCTSNode:
        """
        Expand a new child node or return existing one (assumes expand_lock held).
        
        Must be called from within a 'with self.expand_lock' block to ensure atomicity.
        Handles double-checked locking: if another thread already expanded this move,
        returns the existing child instead of creating a duplicate.
        
        Args:
            move: Move coordinates (row, col).
            player_id: Player ID for new state.
            
        Returns:
            New or existing child node.
        """
        # Double-check pattern: another thread may have expanded this move
        if move in self.children:
            return self.children[move]
        
        # Create and register new child
        new_board = self.board.clone()
        new_board.place_piece(move[0], move[1], self.player_id)
        child = _ParallelMCTSNode(new_board, player_id, parent=self, depth=self.depth + 1)
        self.children[move] = child
        self.reverse_children[id(child)] = move
        return child

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return (
            self.board.is_full()
            or self.board.check_connection(1)
            or self.board.check_connection(2)
        )

    def get_winner(self) -> int:
        """
        Get winner from terminal state.
        
        In Hex, there are no draws: one player must have a winning path.
        
        Returns:
            Winner ID (1 or 2). Returns 0 if unexpectedly not terminal.
        """
        if self.board.check_connection(1):
            return 1
        if self.board.check_connection(2):
            return 2
        # Should not reach here if node is terminal
        return 0
    
    def update_stats(self, result: int, amaf_sequence: List[Tuple[int, int]]) -> None:
        """
        Update statistics for this node based on simulation result with thread safety.
        
        Args:
            result: Player ID of winner (1 or 2).
            amaf_sequence: List of moves played in the simulation for AMAF updates.
        """
        with self.stats_lock:
            self.visit_count += 1
            if result == self.player_id:
                self.win_count += 1
            
            # AMAF statistics from playout sequence
            if self.parent is not None and amaf_sequence:
                for move in amaf_sequence:
                    if move not in self.amaf_visits:
                        self.amaf_visits[move] = 0
                        self.amaf_wins[move] = 0
                    
                    self.amaf_visits[move] += 1
                    if result == self.player_id:
                        self.amaf_wins[move] += 1

class ParallelizedMCTSPlayer(Player):
    """Parallel tree MCTS player for Hex with RAVE/AMAF and tree recycling.
    
    Implements MCTS with:
    
    - Tree Recycling: Between moves, detects opponent's move via board
       comparison and reuses the search tree, improving efficiency.
    
    - RAVE Enhancement: All-Moves-As-First statistics track value estimates
      for moves regardless of their position in the tree. Combines UCT and
      RAVE for faster convergence during early iterations.

    - Tree parallelization: multiple worker threads explore a shared MCTS tree,
      synchronized with fine-grained locks per node. Minimizes contention while 
      maintaining consistency of statistics.
      
      Features:
        - Tree parallelization with customizable amount of worker threads
        - RAVE/AMAF statistics for faster convergence
        - Tree recycling between moves
        - Thread-safe node expansion and backpropagation

    """

    MAX_DEPTH = 50  # Limit depth to prevent infinite loops in large boards
    ALPHA = 0.7  # visit count importance factor for final move selection
    NUM_WORKERS = 4

    def __init__(self, player_id: int, max_time: float = 4.98, exploration_c: float = 0.5, rave_bias: float = 0.00025):
        """
        Initialize parallel MCTS player.
        
        Args:
            player_id: Player ID (1 or 2).
            max_time: Maximum time per move in seconds (default 4.98).
            exploration_c: UCT exploration constant (default 0.5).
                           Controls exploration vs exploitation balance.
                           Higher -> more exploration, lower -> more exploitation.
            rave_bias: Bias for RAVE decay function (default 0.00025).
                       Higher values -> faster transition to UCT only.
                       Lower values -> longer phase of UCT+RAVE combination.
        """
        super().__init__(player_id)
        self.max_time = max_time
        self.exploration_c = exploration_c
        self.rave_bias = rave_bias
        
        # Tree recycling state
        self.board: Optional[BoardOptimized] = None
        self.root: Optional[_ParallelMCTSNode] = None
        self.tree_reused_count = 0  # Statistics: tracks successful tree reuses

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
            root = _ParallelMCTSNode(new_board, self.player_id)

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

        # Parallel MCTS search on shared tree with multiple workers
        iteration_counts = self._parallel_search(root, time_start)

        # Select best move
        best_move = self._select_best_move(root)

        # Save state for next turn's recycling
        self._save_state_for_recycling(new_board, root, best_move)

        elapsed = time.time() - time_start
        total_iterations = sum(iteration_counts)
        print(f"[Player {self.player_id}] Parallel MCTS ({self.NUM_WORKERS} workers) | "
              f"iterations={total_iterations} (per worker: {iteration_counts}) in {elapsed:.4f}s | \n"
              f"c={self.exploration_c:.2f} | bias={self.rave_bias} | "
              f"recycled=x{self.tree_reused_count}")
        
        return best_move

    def _parallel_search(self, root: _ParallelMCTSNode, time_start: float) -> List[int]:
        """
        Execute parallel MCTS search with multiple worker threads on shared tree.
        
        Each worker independently performs MCTS iterations on the same tree,
        with thread-safe synchronization at each node. Workers terminate when
        max_time is reached.
        
        Args:
            root: Root node of shared MCTS tree.
            time_start: Start time for time budget calculation.
        """
        def worker_search(worker_id: int) -> int:
            """Run MCTS iterations until time budget exhausted."""
            iterations = 0
            while time.time() - time_start < self.max_time:
                self._mcts_iteration_shared_tree(root)
                iterations += 1
            return iterations
        
        # Launch workers in parallel
        iteration_counts = [0] * self.NUM_WORKERS
        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
            futures = [executor.submit(worker_search, i) for i in range(self.NUM_WORKERS)]
            iteration_counts = [f.result() for f in futures]
        
        return iteration_counts

    def _mcts_iteration_shared_tree(self, root: _ParallelMCTSNode) -> None:
        """
        Execute one MCTS iteration on shared tree with thread-safe operations.
        
        Phases:
        1. Selection: descend using UCT+RAVE
        2. Expansion: add a new node with untried move
        3. Simulation: random playout with AMAF tracking
        4. Backpropagation: update UCT path and AMAF statistics
        
        Args:
            root: Root node of MCTS tree.
        """
        # Selection + Expansion
        node = root

        while not node.is_terminal() and node.depth < self.MAX_DEPTH:
            with node.expand_lock:
                # Check for untried moves (protected by lock)
                if node.untried_moves:
                    move = node.untried_moves.pop()
                    next_player_id = 3 - node.player_id
                    node = node.expand_and_get_child(move, next_player_id)
                    break
            
            # Selection: use UCT+RAVE (protected by stats_lock inside)
            child = node.select_best_child_with_rave(self.exploration_c, self.rave_bias)
            if child is None:
                break
            node = child

        # Simulation: random playout from node WITH AMAF tracking
        if node.is_terminal():
            result = node.get_winner()
            amaf_sequence = []
        elif len(node.board.get_empty_positions()) < node.board.size * math.sqrt(node.board.size):
            # If we're in the endgame phase (few empty positions), use heuristics to guide playout
            result, amaf_sequence = self._play_endgame_playout_with_heuristics(node)
        else:
            result, amaf_sequence = self._play_random_playout(node)

        # Backpropagation: update UCT path AND AMAF statistics
        current = node
        while current is not None:
            current.update_stats(result, amaf_sequence)
            current = current.parent

    def _play_random_playout(self, node: _ParallelMCTSNode) -> Tuple[int, list]:
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
    
    def _play_endgame_playout_with_heuristics(self, node: _ParallelMCTSNode) -> Tuple[int, list]:
        """
        Play a guided playout from node using endgame heuristics.
        
        Strategy (fallthrough order):
        1. Winning move
        2. Opponent threat block
        3. Connection priority
        4. Random move (fallback)
        
        This produces stronger playouts in endgame while maintaining MCTS purity
        (no modification of statistics, only playout guidance).
        
        Args:
            node: Starting node for playout.
            
        Returns:
            Tuple of (winner, moves_sequence)
        """
        board = node.board.clone()
        current_player = node.player_id
        moves_played = []
        heaps: dict = {current_player: [], 3 - current_player: []}  # Separate heaps for each player

        for move in board.get_empty_positions():
            info = board.move_priority_info(current_player, move)
            # Heuristic: connection priority with max heap
            heappush(heaps[current_player], (-self._evaluate_own_move_priority(info), move))  # Prioritize own connections and block opponent's
            heappush(heaps[3 - current_player], (-self._evaluate_opponent_move_priority(info), move))
        
        while not board.is_full():
            # ← Heuristic 1: WINNING MOVE (O(1), immediate victory)
            winning_move = get_immediate_winning_move(board, current_player)
            if winning_move:
                board.place_piece(winning_move[0], winning_move[1], current_player)
                moves_played.append(winning_move)
                return current_player, moves_played
            
            # ← Heuristic 2: OPPONENT THREAT (O(1), must block)
            threat_move = get_opponent_forcing_move(board, current_player)
            if threat_move:
                board.place_piece(threat_move[0], threat_move[1], current_player)
                moves_played.append(threat_move)
                current_player = 3 - current_player
                continue
            
            # ← Heuristic 3: CONNECTION PRIORITY (O(1) per move, guides toward winning)
            available_moves = board.get_empty_positions()
            best_move = random.choice(list(available_moves))  # Default fallback if heaps are empty

            while heaps[current_player]:
                _, move = heappop(heaps[current_player])
                if move in available_moves: # Checks for validity
                    best_move = move
                    break
            
            board.place_piece(best_move[0], best_move[1], current_player)
            moves_played.append(best_move)
            
            # Check for win after placing
            if board.check_connection(current_player):
                return current_player, moves_played
            
            for nr, nc in board.neighbors(best_move[0], best_move[1]):
                if board.board[nr][nc] == 0:
                    # Update lazy heap
                    info = board.move_priority_info(current_player, (nr, nc))
                    heappush(heaps[current_player], (-self._evaluate_own_move_priority(info), (nr, nc)))
                    heappush(heaps[3 - current_player], (-self._evaluate_opponent_move_priority(info), (nr, nc)))
            
            current_player = 3 - current_player
        
        # Board is full with no winner
        return 0, moves_played

    def _evaluate_own_move_priority(self, info: Tuple[int, int, int, int]) -> int:
        return 5 * info[0] + 10 * info[1] + 15 * info[2] + 20 * info[3]
    
    def _evaluate_opponent_move_priority(self, info: Tuple[int, int, int, int]) -> int:
        return 10 * info[0] + 5 * info[1] + 20 * info[2] + 15 * info[3]

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

    def _find_reusable_root(self, current_board: BoardOptimized) -> Optional[_ParallelMCTSNode]:
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
            reused_node.parent = None  # Detach from old parent to avoid memory issues  
            return reused_node
        else:
            # Move not found in tree (shouldn't happen, but fallback to reset)
            return None

    def _save_state_for_recycling(self, board: BoardOptimized, root: _ParallelMCTSNode, best_move: Tuple[int, int]) -> None:
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
        
    def _select_best_move(self, root: _ParallelMCTSNode) -> Tuple[int, int]:
        """
        Select best move from root using weighted combination of visits and win rate.
        
        Balances visit count (exploration effort) with win rate (exploitation).
        Returns move with highest score: (1-α) * (1 - loss_rate) + α * (visits / total)
        
        Args:
            root: Root node.
            
        Returns:
            Best move (row, col).
        """
        with root.stats_lock:
            if not root.children:
                empty = list(root.board.get_empty_positions())
                return random.choice(empty) if empty else (0, 0) # Fallback to random empty move
            
            total_sims = sum(child.visit_count for child in root.children.values())
            
            def move_score(child: _ParallelMCTSNode) -> float:
                if child.visit_count < 1 or total_sims < 1:
                    return 0.0
                win_rate = 1.0 - child.win_count / child.visit_count
                visit_ratio = child.visit_count / total_sims
                return (1.0 - self.ALPHA) * win_rate + self.ALPHA * visit_ratio
            
            best_child = max(root.children.values(), key=move_score)
            
            # Statistics for logging
            win_rate = 1.0 - best_child.win_count / best_child.visit_count if best_child.visit_count > 0 else 0.0
            
        print(f"[Player {self.player_id}] Best move has {best_child.visit_count} visits and win rate {win_rate:.2f} after {total_sims} simulations")

        # Find move that led to this child
        try:
            return root.reverse_children[id(best_child)]
        except KeyError:
            return (0, 0)
