"""Game phase detection and parameter adaptation for MCTS."""

from __future__ import annotations
from typing import Dict, Optional, Any


class GamePhaseManager:
    """
    Manages game phase transitions based on MCTS search progression.
    
    Key principle: Phase is determined by node visit count (iterations performed).

    Based on: Chaslot et al. (2008) "Progressive Strategies for Monte-Carlo Tree Search"
    - Section 3.1: Progressive Bias
    
    "A progressive strategy is similar to a simulation strategy when a few games 
    have been played, and converges to a selection strategy when numerous games 
    have been played."
    
    Phases (based on visit count at each node):
    - OPENING (0-30 visits):     Simulation strategy dominates, heuristics guide
    - MIDGAME (30-100 visits):    Transition between simulation and selection
    - ENDGAME (100+ visits):      Selection strategy dominates, UCT pure
    """
    
    # Visit count thresholds for phase transitions (calibrated for Hex, similar to Mango Go)
    _OPENING_THRESHOLD = 30      # Below: simulation strategy dominates
    _ENDGAME_THRESHOLD = 100     # Above: selection strategy dominates
    
    # Adaptive parameters per game phase
    _PHASE_PARAMETERS = {
        "OPENING": {
            "exploration_c": 0.7,         # Increased exploration in opening phase
            "rave_bias": 0.0001,          # RAVE dominates longer
            "max_depth": 50,              # No practical limit (high value)
            "use_heuristics": False,      # Heuristics not reliable in opening
        },
        "MIDGAME": {
            "exploration_c": 0.4,         # Standard balance (Cazenave empirically optimal)
            "rave_bias": 0.00025,         # RAVE and UCT balanced
            "max_depth": 50,              # Standard search depth
            "use_heuristics": False,      # Heuristics still unreliable
        },
        "ENDGAME": {
            "exploration_c": 0.1,         # Deep exploitation phase
            "rave_bias": 0.001,           # Rapid transition to pure UCT
            "max_depth": 100,             # Deeper search (naturally small tree)
            "use_heuristics": True,       # Deterministic threat detection enabled
        },
    }
    
    @staticmethod
    def get_phase(visit_count: int) -> str:
        """
        Detect current game phase based on node visit count (iterations).
        
        Args:
            visit_count: Number of simulations/visits through this node (int >= 0)
            
        Returns:
            Phase string: "OPENING" | "MIDGAME" | "ENDGAME"
        """
        if visit_count < GamePhaseManager._OPENING_THRESHOLD:
            return "OPENING"
        elif visit_count < GamePhaseManager._ENDGAME_THRESHOLD:
            return "MIDGAME"
        else:
            return "ENDGAME"
    
    @staticmethod
    def get_parameters(phase: str) -> Dict[str, Any]:
        """
        Retrieve adaptive RAVE parameters for the given game phase.
        
        Progressive strategies adapt parameters based on search maturity:
        
        1. exploration_c (UCT exploration-exploitation balance):
           - OPENING: Higher exploration in early iterations (simulation-dominant)
           - MIDGAME: Balanced transition as selection strategy grows stronger
           - ENDGAME: Deep exploitation in selection-dominant phase
        
        2. rave_bias (Controls transition speed from RAVE-dominant to UCT-dominant):
           - OPENING: Low bias = RAVE dominates (simulation guidance)
           - MIDGAME: Intermediate bias = gradual transition
           - ENDGAME: High bias = rapid convergence to pure UCT
        
        3. use_heuristics (based on Lanctot 2014):
           - OPENING/MIDGAME: False (heuristics unreliable with many alternatives)
           - ENDGAME: True (deterministic threat detection effective)
        
        Args:
            phase: Phase identifier - "OPENING" | "MIDGAME" | "ENDGAME"
            
        Returns:
            Dictionary containing adaptive RAVE parameters for the phase
            
        Raises:
            ValueError: If phase is not one of the valid options
        """
        if phase not in GamePhaseManager._PHASE_PARAMETERS:
            raise ValueError(f"Invalid phase: {phase}. Must be OPENING, MIDGAME, or ENDGAME")
        
        return GamePhaseManager._PHASE_PARAMETERS[phase]
    
    @staticmethod
    def log_phase_info(phase: str, visit_count: int) -> str:
        """
        Generate formatted log message for phase transition events.
        
        Args:
            phase: Current game phase ("OPENING" | "MIDGAME" | "ENDGAME")
            visit_count: Number of visits/simulations through current node
            
        Returns:
            Formatted string with phase and visit count information
        """
        return (
            f"[Phase] {phase.upper()} (visits={visit_count})"
        )
