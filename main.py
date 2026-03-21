"""Main module for testing Hex players."""

from __future__ import annotations

import sys
from board import HexBoard
from players.player import Player
from players.random_player import RandomPlayer
from players.basic_MCTS_player import BasicMCTSPlayer
from players.recycler_MCTS_player import RecyclerMCTSPlayer
from players.RAVE_MCTS_player import RAVEMCTSPlayer
from players.progressive_MCTS_player import ProgressiveMCTSPlayer
from players.parallelized_MCTS_player import ParallelizedMCTSPlayer
from typing import Optional, Type


# Map of available player names to classes
AVAILABLE_PLAYERS = {
    "random": RandomPlayer,
    "mcts": BasicMCTSPlayer,
    "mcts-recycler": RecyclerMCTSPlayer,
    "mcts-rave": RAVEMCTSPlayer,
    "mcts-progressive": ProgressiveMCTSPlayer,
    "mcts-parallel": ParallelizedMCTSPlayer,
}


def get_player_class(player_name: str) -> Optional[Type[Player]]:
    """
    Get player class by name.
    
    Args:
        player_name: Name of the player (case-insensitive).
        
    Returns:
        Player class if found, None otherwise.
    """
    return AVAILABLE_PLAYERS.get(player_name.lower())


def display_board(board: HexBoard, move: Optional[tuple] = None) -> None:
    """
    Display the current board state.
    
    Args:
        board: HexBoard instance to display.
        move: Optional last move to highlight.
    """
    print()
    for r in range(board.size):
        indent = " " * (r%2)
        row_symbols = []
        for c in range(board.size):
            value = board.board[r][c]
            if move and move == (r, c):
                row_symbols.append("*" if value != 0 else "*")
            else:
                row_symbols.append("." if value == 0 else str(value))
        print(indent + " ".join(row_symbols))
    print()


def play_game(board_size: int, player1_class, player2_class, verbose: bool = True) -> int:
    """
    Play a complete game between two players.
    
    Args:
        board_size: Size of the hexagonal board.
        player1_class: Class for player 1 (horizontal).
        player2_class: Class for player 2 (vertical).
        verbose: If True, display board state and moves.
        
    Returns:
        Winning player ID (1 or 2).
    """
    board = HexBoard(board_size)
    player1 = player1_class(player_id=1)
    player2 = player2_class(player_id=2)
    players = [None, player1, player2]
    
    current_player = 1
    move_count = 0
    max_moves = board_size * board_size
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game started: {board_size}x{board_size} board")
        print(f"Player 1 ({player1.__class__.__name__}): Horizontal (left-right)")
        print(f"Player 2 ({player2.__class__.__name__}): Vertical (top-bottom)")
        print(f"{'='*60}")
        display_board(board)
    
    while move_count < max_moves:
        # Get move from current player
        move = players[current_player].play(board)
        
        # Place piece
        if not board.place_piece(move[0], move[1], current_player):
            if verbose:
                print(f"❌ Player {current_player} made invalid move: {move}")
            return 3 - current_player  # Other player wins by default
        
        move_count += 1
        
        if verbose:
            player_name = players[current_player].__class__.__name__
            print(f"Move {move_count}: Player {current_player} ({player_name}) plays {move}")
            display_board(board, move)
        
        # Check for winner
        if board.check_connection(current_player):
            if verbose:
                print(f"\n🎉 Player {current_player} wins!")
                print(f"{'='*60}\n")
            return current_player
        
        # Switch player
        current_player = 3 - current_player
    
    # Board is full, no winner (shouldn't happen in Hex)
    if verbose:
        print(f"\n⚠️ Board is full, no winner determined")
        print(f"{'='*60}\n")
    return 0


def run_matches(
    board_size: int,
    player1_class: Type[Player],
    player2_class: Type[Player],
    num_matches: int = 5
) -> None:
    """
    Run multiple matches between two player types.
    
    Args:
        board_size: Size of the hexagonal board.
        player1_class: Class for player 1.
        player2_class: Class for player 2.
        num_matches: Number of matches to play.
    """
    player1_name = player1_class.__name__
    player2_name = player2_class.__name__
    
    print(f"\n{'#'*60}")
    print(f"Testing: {player1_name} vs {player2_name}")
    print(f"{num_matches} matches on {board_size}x{board_size} board")
    print(f"{'#'*60}\n")
    
    player1_wins = 0
    player2_wins = 0
    
    for match_num in range(1, num_matches + 1):
        print(f"Match {match_num}/{num_matches}:")
        print("-" * 60)
        
        # Alternate which player goes first
        if match_num % 2 == 1:
            p1_class = player1_class
            p2_class = player2_class
        else:
            p1_class = player2_class
            p2_class = player1_class
        
        winner = play_game(board_size, p1_class, p2_class, verbose=False)
        
        if winner == 1:
            print(f"✅ Player 1 ({p1_class.__name__}) wins!")
            if p1_class == player1_class:
                player1_wins += 1
            else:
                player2_wins += 1
        elif winner == 2:
            print(f"✅ Player 2 ({p2_class.__name__}) wins!")
            if p2_class == player2_class:
                player2_wins += 1
            else:
                player1_wins += 1
        else:
            print(f"⚠️ Draw")
        print()
    
    print(f"{'#'*60}")
    print(f"Results: {player1_name}: {player1_wins}, {player2_name}: {player2_wins}")
    print(f"Win rate: {player1_name}: {100*player1_wins/num_matches:.1f}%")
    print(f"{'#'*60}\n")


def display_help() -> None:
    """Display usage information and available players."""
    print(f"\n{'='*60}")
    print("Hex Player Test Platform")
    print(f"{'='*60}")
    print("\nUsage:")
    print("  python main.py [board_size] [player1] [player2] [num_matches]")
    print("\nArguments:")
    print("  board_size   - Size of hexagonal board (default: 5)")
    print("  player1      - First player type (default: random)")
    print("  player2      - Second player type (default: mcts)")
    print("  num_matches  - Number of matches (default: 1)")
    print("               - If 1: verbose single game")
    print("               - If >1: silent mode with statistics")
    print("\nAvailable players:")
    for name in AVAILABLE_PLAYERS.keys():
        print(f"  {name}")
    print("\nExamples:")
    print("  python main.py                         # 5x5, random vs mcts, 1 game (verbose)")
    print("  python main.py 4                       # 4x4, random vs mcts, 1 game (verbose)")
    print("  python main.py 5 mcts mcts-recycler 5  # 5x5, mcts vs mcts-recycler, 5 games (stats)")
    print("  python main.py 5 mcts-recycler mcts-rave 5  # mcts-recycler vs RAVE (Phase 3)")
    print("  python main.py 5 mcts-rave mcts-progressive 5  # RAVE vs Progressive (Phase 4)")
    print("  python main.py 3 mcts-rave random 10   # 3x3, RAVE vs random, 10 games")
    print("  python main.py 5 mcts-rave mcts-parallel 5   # 3x3, RAVE vs parallel, 5 games")
    print(f"{'='*60}\n")


def main() -> None:
    """Main entry point with flexible player selection."""
    # Parse arguments
    board_size = 5
    player1_name = "random"
    player2_name = "mcts"
    num_matches = 1
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h", "help"]:
            display_help()
            return
        try:
            board_size = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid board size: {sys.argv[1]}")
            display_help()
            return
    
    if len(sys.argv) > 2:
        player1_name = sys.argv[2]
    
    if len(sys.argv) > 3:
        player2_name = sys.argv[3]
    
    if len(sys.argv) > 4:
        try:
            num_matches = int(sys.argv[4])
        except ValueError:
            print(f"❌ Invalid number of matches: {sys.argv[4]}")
            display_help()
            return
    
    # Get player classes
    player1_class = get_player_class(player1_name)
    player2_class = get_player_class(player2_name)
    
    # Validate players
    if player1_class is None:
        print(f"❌ Player not found: {player1_name}")
        print(f"Available players: {', '.join(AVAILABLE_PLAYERS.keys())}")
        return
    
    if player2_class is None:
        print(f"❌ Player not found: {player2_name}")
        print(f"Available players: {', '.join(AVAILABLE_PLAYERS.keys())}")
        return
    
    # Validate board size
    if not (0 < board_size <= 25):
        print(f"❌ Invalid board size: {board_size}. Must be between 1 and 25")
        return
    
    # Run game(s)
    if num_matches == 1:
        # Single verbose game
        print(f"\nSingle game mode - verbose output:")
        play_game(board_size, player1_class, player2_class, verbose=True)
    else:
        # Multiple matches with statistics
        run_matches(board_size, player1_class, player2_class, num_matches)


if __name__ == "__main__":
    main()
