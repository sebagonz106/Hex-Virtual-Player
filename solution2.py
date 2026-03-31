"""RAVE MCTS player for Hex game with tree parallelization.

Implements tree parallelization where multiple worker threads explore a shared
MCTS tree simultaneously. Each node has fine-grained locks to synchronize
statistic updates while minimizing contention. Includes helper methods and classes
such as ParallelMCTSNode for thread-safe node management and BoardOptimized for
efficient board state representation with Union-Find. Also features tree recycling.
"""

from __future__ import annotations
import time
import random
import math
import threading
from typing import Optional, Tuple, List, Set
from heapq import heappush, heappop
from concurrent.futures import ThreadPoolExecutor

from board import HexBoard
from players.player import Player

class _MCTSNode:
    """Node in the MCTS tree with thread-safe RAVE/AMAF statistics.
    
    Uses fine-grained locking: each node has independent locks for statistics
    updates and child expansion. This minimizes lock contention in a shared tree.
    """

    def __init__(
        self,
        board: _BoardOptimized,
        player_id: int,
        parent: Optional[_MCTSNode] = None,
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
        self.parent: Optional[_MCTSNode] = parent
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

    def select_best_child_with_rave(self, exploration_c: float, rave_bias: float = 0.00025) -> Optional[_MCTSNode]:
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
            
            def combined_value(child: _MCTSNode) -> float:
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
            if empty < self.board.size * math.sqrt(self.board.size):
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

    def expand_and_get_child(self, move: Tuple[int, int], player_id: int) -> _MCTSNode:
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
        child = _MCTSNode(new_board, player_id, parent=self, depth=self.depth + 1)
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



class SmartPlayer2(Player):
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
    HEURISTICS_SELECTION_RATE = 0.9  # Probability of using heuristics for move selection in late game

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
        self.board: Optional[_BoardOptimized] = None
        self.root: Optional[_MCTSNode] = None
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
        # print("========================== player", self.player_id, "==========================")
        
        time_start = time.time()

        # Convert to optimized board immediately for all subsequent operations
        new_board = _BoardOptimized(board)

        best_move = random.choice(list(new_board.get_empty_positions()))  # Fallback random move

        # Early check: opening move suggestion
        opening_move = suggest_opening_move(new_board, self.player_id)
        if opening_move and new_board.board[opening_move[0]][opening_move[1]] == 0:
            self._reset_info()
            return opening_move
        
        # Early check: immediate winning move
        win_move = get_immediate_winning_move(new_board, self.player_id)
        if win_move:
            self._reset_info()
            return win_move

        # Attempt tree recycling
        root = self._find_reusable_root(new_board)
        recycled = False
        if root is None or not self.board:
            # No recycling possible, create fresh root
            root = _MCTSNode(new_board, self.player_id)
            self.board = None
        else:
            new_board = self.board.clone()
            recycled = True

        # Early check: must block opponent
        opponent_id = 3 - self.player_id
        block_move = get_opponent_forcing_move(new_board, opponent_id)
        if block_move:
            self._save_state_for_recycling(new_board, root, block_move)
            return block_move

        # Early check: bridge disruption
        if recycled:
            forced_moves = new_board.get_altered_bridges()
            if len(forced_moves) > 0:
                move = forced_moves[0]

                # several moves affected, choose one using selection criteria
                if len(forced_moves) > 1:
                    nodes = []
                    for move in forced_moves:
                        try:
                            nodes.append(root.children[move])
                        except KeyError:
                            continue

                    if len(nodes) > 0:
                        move = root.reverse_children[id(max(nodes, key=self._move_score))]
                self._save_state_for_recycling(new_board, root, move)
                return move

        # Parallel MCTS search on shared tree with multiple workers
        iteration_counts = self._parallel_search(root, time_start)

        # Select best move
        best_move = self._select_best_move(root)

        # Save state for next turn's recycling
        self._save_state_for_recycling(new_board, root, best_move)

        best_child = root.children[best_move]
        # print("tree recycled:", recycled)
        # print("best move:", best_move)
        # print("sims:", root.visit_count)
        # print("selected visit_count:", best_child.visit_count)
        # print("max visit_count:", max(child.visit_count for child in root.children.values()))
        # print("selected win_rate:", 1 - best_child.win_count / best_child.visit_count)
        # print("max win_rate:", max( 1 - child.win_count / child.visit_count for child in root.children.values()))
        # print("elapsed:", time.time() - time_start)
        # print("===============================================================")
        
        return best_move

    def _parallel_search(self, root: _MCTSNode, time_start: float) -> List[int]:
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

    def _mcts_iteration_shared_tree(self, root: _MCTSNode) -> None:
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

        u = random.uniform(0, 1)

        # Simulation: random playout from node WITH AMAF tracking
        if node.is_terminal():
            result = node.get_winner()
            amaf_sequence = []
        elif len(node.board.get_empty_positions()) < node.board.size * math.sqrt(node.board.size) and u < self.HEURISTICS_SELECTION_RATE:
            # If we're in the endgame phase (few empty positions), use heuristics to guide playout
            result, amaf_sequence = self._play_endgame_playout_with_heuristics(node)
        else:
            result, amaf_sequence = self._play_random_playout(node)

        # Backpropagation: update UCT path AND AMAF statistics
        current = node
        while current is not None:
            current.update_stats(result, amaf_sequence)
            current = current.parent

    def _play_random_playout(self, node: _MCTSNode) -> Tuple[int, list]:
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
    
    def _play_endgame_playout_with_heuristics(self, node: _MCTSNode) -> Tuple[int, list]:
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

    def _find_board_difference(self, board1: _BoardOptimized, board2: _BoardOptimized) -> Optional[Tuple[int, int]]:
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

    def _find_reusable_root(self, current_board: _BoardOptimized) -> Optional[_MCTSNode]:
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
            self.board.place_piece(opp_move[0], opp_move[1], 3 - self.player_id)
            reused_node.parent = None  # Detach from old parent to avoid memory issues  
            return reused_node
        else:
            # Move not found in tree (shouldn't happen, but fallback to reset)
            return None

    def _save_state_for_recycling(self, board: _BoardOptimized, root: _MCTSNode, best_move: Tuple[int, int]) -> None:
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
        saved_board.update_bridges(best_move)

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
        
    def _select_best_move(self, root: _MCTSNode) -> Tuple[int, int]:
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
            
            
            
            best_move = root.reverse_children[id(max(root.children.values(), key=self._move_score))]
        
        # Find move that led to this child
        try:
            return best_move
        except KeyError:
            return (0, 0)

    def _move_score(self, child: _MCTSNode) -> float:
        """Calculate a move score from a node based on visit count and win rate for final move selection.

        Args:
            child (_MCTSNode): Node from which to calculate the move score.

        Returns:
            float: Move score based on visit count and win rate.
        """
        root = child.parent

        if root is None:
            return 0.0

        total_sims = sum(child.visit_count for child in root.children.values())

        if child.visit_count < 1 or total_sims < 1:
            return 0.0
        
        win_rate = 1.0 - child.win_count / child.visit_count
        visit_ratio = child.visit_count / total_sims
        return (1.0 - self.ALPHA) * win_rate + self.ALPHA * visit_ratio
    

class _UnionSnapshot:
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


class _Bridge:
    """Represents a bridge connection pattern in Hex.
    
    A bridge is a fundamental connection structure where two player pieces
    are connected through two empty cells (connectors). If one connector is
    blocked by opponent, the player must occupy the other to maintain potential
    connection.
    
    In game theory, bridges are used to track forced moves and strategic
    connections that guarantee at least one path remains open.
    """
    
    def __init__(self, player_id: int, positions: set[Tuple[int, int]], connectors: set[Tuple[int, int]]) -> None:
        """Initialize a bridge structure.
        
        Args:
            player_id: Owner of the bridge (1 or 2).
            positions: Set of two player pieces forming the bridge endpoints.
            connectors: Set of two empty cells connecting the positions.
                       If one is blocked, the other becomes a forced move.
        """
        self.player_id = player_id
        self.positions = positions
        self.connectors = connectors

    def __eq__(self, value: object) -> bool:
        """Check equality based on positions and connectors.
        
        Two bridges are equal if they have the same endpoints and connectors,
        regardless of internal set ordering.
        
        Args:
            value: Object to compare with.
            
        Returns:
            True if bridges represent the same structure, False otherwise.
        """
        if not isinstance(value, _Bridge) or self.player_id != value.player_id:
            return False
        
        # Verify all positions match
        for pos in value.positions:
            if pos not in self.positions:
                return False
        
        # Verify all connectors match
        for pos in value.connectors:
            if pos not in self.connectors:
                return False
            
        return True


class _BoardOptimized:
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
        self._bridges: List[_Bridge] = []

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
        self.move_union_stack: List[List[_UnionSnapshot]] = []
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

    def _union(self, x: int, y: int) -> Optional[_UnionSnapshot]:
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
            snapshot = _UnionSnapshot(x_root, self.parent[x_root], self.rank[x_root])
            self.parent[x_root] = y_root
        else:
            snapshot = _UnionSnapshot(y_root, self.parent[y_root], self.rank[y_root])
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
                    for nr, nc in self.neighbors(r, c):
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
        unions_in_move: List[_UnionSnapshot] = []

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
        for nr, nc in self.neighbors(row, col):
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

    def neighbors(self, row: int, col: int) -> list:
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
    
    def move_priority_info(self, player: int, pos: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Information needed for move priority analysis based on connection potential for both players.

        Args:
            board (BoardOptimized): board
            player (int): player
            pos (Tuple[int, int]): position

        Returns:
            (int, int, int, int):  (our_neighbors, opp_neighbors, our_bridges, opp_bridges)
        """

        if(self.board[pos[0]][pos[1]] != 0):
            return 0, 0, 0, 0 # Invalid move, no priority

        our_neighbors = 0
        opp_neighbors = 0
        our_bridges = 0
        opp_bridges = 0

        empty_neighbours = []

        for nr, nc in self.neighbors(pos[0], pos[1]):
            if self.board[nr][nc] == player:
                our_neighbors += 1
            elif self.board[nr][nc] == 3 - player:
                opp_neighbors += 1
            else:
                empty_neighbours.append((nr, nc))

        bridges_found = []

        if len(empty_neighbours) > 1:
            for r, c in empty_neighbours:
                if (r, c) in bridges_found: #avoid double count
                    continue
                
                # Exclude original pos
                current_neighbours = [(x, y) for x, y in self.neighbors(r, c) if not (x == pos[0] and y == pos[1])]

                for nr, nc in current_neighbours:
                    if not (nr, nc) in empty_neighbours:
                        continue
                    # Get common neighbour other than pos
                    candidates = [(x,y) for (x,y) in current_neighbours if (x, y) in self.neighbors(nr, nc)]
                    
                    if len(candidates) > 0: 
                        x, y = candidates[0] # Just one possible candidate
                        bridges_found.append((nr, nc))
                        if self.board[x][y] == player:
                            our_bridges += 1
                        else:
                            opp_bridges += 1 # Enemy bridge or possible bridge (if 0)

        return opp_neighbors, our_neighbors, opp_bridges, our_bridges #connection heuristic data
    

    def update_bridges(self, pos: Tuple[int, int]) -> None:
        """Detect and register bridge patterns created by a placed piece.
        
        When a piece is placed, identifies all bridge structures involving the
        new piece. A bridge connects the new piece to another friendly piece
        through exactly two empty intermediate cells.
        
        Bridge structure:
            Piece (new) -- empty cell -- empty cell -- Piece (existing)
        
        For each identified bridge, both connectors are tracked. If one is
        later blocked by the opponent, occupying the other becomes a forced
        move to maintain the connection guarantee.
        
        Complexity: O(K²) where K ~ 6 (hexagonal grid neighbors).
        
        Args:
            pos: Position (row, col) where piece was just placed.
        """
        # Get player who placed the piece
        player = self.board[pos[0]][pos[1]]
        if player == 0:  # Invalid: empty cell (should not happen)
            return
        
        # Find all empty neighbors of the placed piece
        empty_neighbours = [(r, c) for r, c in self.neighbors(pos[0], pos[1]) 
                           if self.board[r][c] == 0]

        # Need at least 2 empty neighbors to form a bridge
        if len(empty_neighbours) < 2:
            return
        
        # Check each pair of empty neighbors for bridge pattern
        for r, c in empty_neighbours:
            # Get neighbors of this empty cell (excluding the piece we just placed)
            current_neighbours = [(x, y) for x, y in self.neighbors(r, c) 
                                 if not (x == pos[0] and y == pos[1])]

            for nr, nc in current_neighbours:
                # Other connector must also be an empty neighbor of placed piece
                if (nr, nc) not in empty_neighbours:
                    continue
                    
                # Find the other friendly piece this bridge connects to
                candidates = [(x, y) for x, y in current_neighbours 
                             if (x, y) in self.neighbors(nr, nc) 
                             and self.board[x][y] == player]
                
                if candidates:
                    # Create and register the bridge
                    other_piece = candidates[0]
                    bridge = _Bridge(player, set([pos, other_piece]), 
                                   set([(r, c), (nr, nc)]))
                    
                    # Avoid registering duplicates
                    if bridge not in self._bridges:
                        self._bridges.append(bridge)
    
    def get_altered_bridges(self) -> list[Tuple[int, int]]:
        """Identify forced moves derived from compromised or broken bridges.
        
        Scans all tracked bridges to detect when opponent has blocked one
        connector. When exactly one connector remains free, that position
        becomes a forced move to preserve the connection.
        
        Also performs cleanup by removing bridges with both connectors blocked.
        
        Strategy (per bridge):
        - 2 connectors free -> viable, no forced move
        - 1 connector free  -> forced move to occupy the remaining connector
        - 0 connectors free -> broken, remove from tracking
        
        Complexity: O(B) where B is the number of tracked bridges.
        
        Returns:
            List of forced move positions (row, col) defending critical bridges.
            Empty list if no compromised bridges exist.
        """
        forced_moves = []
        bridges_to_remove = []

        # Check each tracked bridge for blockage
        for bridge in self._bridges:
            # Count free (empty) connectors for this bridge
            free_connectors = [(r, c) for r, c in bridge.connectors 
                              if self.board[r][c] == 0]
            
            if len(free_connectors) > 1:
                # Bridge still viable: both paths remain open
                continue
            elif len(free_connectors) > 0:
                # Bridge compromised: one path blocked, occupy the other
                forced_moves.append(free_connectors[0])
            else:
                # Bridge destroyed: both paths blocked, mark for removal
                bridges_to_remove.append(bridge)
        
        # Perform cleanup of broken bridges
        for bridge in bridges_to_remove:
            self._bridges.remove(bridge)
        
        return forced_moves

    def clone(self) -> "_BoardOptimized":
        """
        Create a deep copy of this board with complete state.
        
        Includes Union-Find structure and move history.
        """
        new_board = _BoardOptimized.__new__(_BoardOptimized)
        new_board.size = self.size
        new_board.board = [row[:] for row in self.board]
        new_board._empty_positions = self._empty_positions.copy()
        new_board._neighbors_cache = self._neighbors_cache
        new_board._bridges = self._bridges.copy()

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
            [_UnionSnapshot(s.node_idx, s.prev_parent, s.prev_rank) for s in move]
            for move in self.move_union_stack
        ]
        new_board.move_history = self.move_history.copy()

        return new_board


# ================== Early move detection for immediate tactical gains. ==================


def get_immediate_winning_move(board: _BoardOptimized, player_id: int) -> Optional[Tuple[int, int]]:
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
                for nr, nc in board.neighbors(r, c):
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


def get_opponent_forcing_move(board: _BoardOptimized, opp_id: int) -> Optional[Tuple[int, int]]:
    """
    Detect if opponent has a winning move. Must block it.
    
    Args:
        board: BoardOptimized instance.
        opp_id: Opponent's player ID (1 or 2).
        
    Returns:
        Opponent's winning move to block (row, col) or None.
    """
    return get_immediate_winning_move(board, opp_id)


def suggest_opening_move(board: _BoardOptimized, player_id: int) -> Optional[Tuple[int, int]]:
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
        neighbors = list(board.neighbors(center_r, center_c))
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